%{
//#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <libgen.h> // dirname

#include "ast.h"
#include "codegen.h"

#define YYSTYPE ASTNode*

ASTNode* root = NULL;

extern FILE* yyin;
extern char* yytext;

int yylex();
int yyparse();
int yyerror(const char* str);
int yyget_lineno();

void
cleanup(void)
{
    if (root)
        astnode_destroy(root); // Frees all children and itself
}

void set_identifier_type(const NodeType type, ASTNode* curr);
void set_identifier_prefix(const char* prefix, ASTNode* curr);
void set_identifier_infix(const char* infix, ASTNode* curr);
ASTNode* get_node(const NodeType type, ASTNode* node);
ASTNode* get_node_by_token(const int token, ASTNode* node);

void
process_includes(const size_t depth, const char* dir, const char* file, FILE* out)
{
  const size_t max_nests = 64;
  if (depth >= max_nests) {
    fprintf(stderr, "CRITICAL ERROR: Max nests %lu reached when processing includes. Aborting to avoid thrashing the disk. Possible reason: circular includes.\n", max_nests);
    exit(EXIT_FAILURE);
  }

  printf("Building AC object %s\n", file);
  FILE* in = fopen(file, "r");
  if (!in) {
    fprintf(stderr, "FATAL ERROR: could not open include file '%s'\n", file);
    assert(in);
  }

  const size_t  len = 4096;
  char buf[len];
  while (fgets(buf, len, in)) {
    char* line = buf;
    while (strlen(line) > 0 && line[0] == ' ') // Remove whitespace
      ++line;

    if (!strncmp(line, "#include", strlen("#include"))) {

      char incl[len];
      sscanf(line, "#include \"%[^\"]\"\n", incl);

      char path[len];
      sprintf(path, "%s/%s", dir, incl);

      fprintf(out, "// Include file %s start\n", path);
      process_includes(depth+1, dir, path, out);
      fprintf(out, "// Included file %s end\n", path);

    } else {
      fprintf(out, "%s", buf);
    }
  }

  fclose(in);
}

void
process_hostdefines(const char* file_in, const char* file_out)
{
  FILE* in = fopen(file_in, "r");
  assert(in);

  FILE* out = fopen(file_out, "w");
  assert(out);

  const size_t  len = 4096;
  char buf[len];
  while (fgets(buf, len, in)) {
    fprintf(out, "%s", buf);

    char* line = buf;
    while (strlen(line) > 0 && line[0] == ' ') // Remove whitespace
      ++line;

    if (!strncmp(line, "hostdefine", strlen("hostdefine"))) {
      while (strlen(line) > 0 && line[0] != ' ') // Until whitespace
        ++line;

      fprintf(out, "#define%s", line);
    }
  }

  fclose(in);
  fclose(out);
}

void
format_source(const char* file_in, const char* file_out)
{
   FILE* in = fopen(file_in, "r");
  assert(in);

  FILE* out = fopen(file_out, "w");
  assert(out);

  while (!feof(in)) {
    const char c = fgetc(in);
    if (c == EOF)
      break;

    fprintf(out, "%c", c);
    if (c == ';')
      fprintf(out, "\n");
  }

  fclose(in);
  fclose(out);
}

int code_generation_pass(const char* stage0, const char* stage1, const char* stage2, const char* stage3, const char* dir, const bool gen_mem_accesses)
{
        // Stage 0: Clear all generated files to ensure acc failure can be detected later
        {
          const char* files[] = {"user_declarations.h", "user_defines.h", "user_kernels.h"};
          for (size_t i = 0; i < sizeof(files)/sizeof(files[0]); ++i) {
            FILE* fp = fopen(files[i], "w");
            assert(fp);
            fclose(fp);
          }
        }

        // Stage 1: Preprocess includes
        {
          FILE* out = fopen(stage1, "w");
          assert(out);
        
          process_includes(0, dir, stage0, out);

          fclose(out);
        }

        // Stage 2: Preprocess hostdefines
        {
          process_hostdefines(stage1, stage2);
        }

        // Stage 3: Preprocess everything else
        {
          const size_t cmdlen = 4096;
          char cmd[cmdlen];
          snprintf(cmd, cmdlen, "gcc -x c -E %s > %s", stage2, stage3);
          const int retval = system(cmd);
          if (retval == -1) {
              fprintf(stderr, "Catastrophic error: preprocessing failed.\n");
              assert(retval != -1);
          }
        }

        // Generate code
        yyin = fopen(stage3, "r");
        if (!yyin)
            return EXIT_FAILURE;

        int error = yyparse();
        if (error)
            return EXIT_FAILURE;

        // generate(root, stdout);
        FILE* fp = fopen("user_kernels.h.raw", "w");
        assert(fp);
        generate(root, fp, gen_mem_accesses);
        fclose(fp);

        fclose(yyin);

        // Stage 4: Format
        format_source("user_kernels.h.raw", "user_kernels.h");

        return EXIT_SUCCESS;
}

int
main(int argc, char** argv)
{
    atexit(&cleanup);

    if (argc > 2) {
      fprintf(stderr, "Error multiple .ac files passed to acc, can only process one at a time. Ensure that DSL_MODULE_DIR contains only one .ac file.\n");
      return EXIT_FAILURE;
    }

    if (argc == 2) {

        char stage0[strlen(argv[1])];
        strcpy(stage0, argv[1]);
        const char* stage1 = "user_kernels.ac.pp_stage1";
        const char* stage2 = "user_kernels.ac.pp_stage2";
        const char* stage3 = "user_kernels.ac.pp_stage3";
        const char* dir = dirname(argv[1]); // WARNING: dirname has side effects!

        if (OPTIMIZE_MEM_ACCESSES) {
          code_generation_pass(stage0, stage1, stage2, stage3, dir, true); // Uncomment to enable stencil mem access checking
          generate_mem_accesses(); // Uncomment to enable stencil mem access checking
        }
        code_generation_pass(stage0, stage1, stage2, stage3, dir, false);
        

        return EXIT_SUCCESS;
    } else {
        puts("Usage: ./acc [source file]");
        return EXIT_FAILURE;
    }
}
%}

%token IDENTIFIER STRING NUMBER REALNUMBER DOUBLENUMBER
%token IF ELIF ELSE WHILE FOR RETURN IN BREAK CONTINUE
%token BINARY_OP ASSIGNOP
%token INT UINT INT3 REAL REAL3 MATRIX FIELD STENCIL
%token KERNEL SUM MAX
%token HOSTDEFINE

%%

root: program { root = astnode_create(NODE_UNKNOWN, $1, NULL); }
    ;

program: /* Empty*/                  { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); }
       | program variable_definition {
            $$ = astnode_create(NODE_UNKNOWN, $1, $2);

            ASTNode* variable_definition = $$->rhs;
            assert(variable_definition);

            ASTNode* declaration = get_node(NODE_DECLARATION, variable_definition);
            assert(declaration);

            ASTNode* declaration_list = declaration->rhs;
            assert(declaration_list);

            const ASTNode* is_field = get_node_by_token(FIELD, $$->rhs);
            if (is_field) {
                variable_definition->type |= NODE_FIELD;
                set_identifier_type(NODE_FIELD_ID, declaration_list);
            } else {
                variable_definition->type |= NODE_DCONST;
                set_identifier_type(NODE_DCONST_ID, declaration_list);
            }

            ASTNode* assignment = get_node(NODE_ASSIGNMENT, variable_definition);
            if (assignment) {
                fprintf(stderr, "FATAL ERROR: Device constant assignment is not supported. Load the value at runtime with ac[Grid|Device]Load[Int|Int3|Real|Real3]Uniform-type API functions or use #define.\n");
                assert(!assignment);
            }
         }
       | program function_definition { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       | program stencil_definition  { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       | program hostdefine          { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       ;

/*
 * =============================================================================
 * Terminals
 * =============================================================================
 */
identifier: IDENTIFIER { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
number: NUMBER         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
      | REALNUMBER     { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_prefix("AcReal(", $$); astnode_set_postfix(")", $$); }
      | DOUBLENUMBER   {
            $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken;
            astnode_set_prefix("double(", $$); astnode_set_postfix(")", $$);
            $$->buffer[strlen($$->buffer) - 1] = '\0'; // Drop the 'd' postfix
        }
      ;
string: STRING         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
if: IF                 { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
elif: ELIF             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
else: ELSE             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
while: WHILE           { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
for: FOR               { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
in: IN                 { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; };
int: INT               { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
uint: UINT             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
int3: INT3             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
real: REAL             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("AcReal", $$); /* astnode_set_buffer(yytext, $$); */ $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
real3: REAL3           { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("AcReal3", $$); /* astnode_set_buffer(yytext, $$); */ $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
matrix: MATRIX         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("AcMatrix", $$); /* astnode_set_buffer(yytext, $$); */ $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
field: FIELD           { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
stencil: STENCIL       { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("", $$); /*astnode_set_buffer(yytext, $$);*/ $$->token = 255 + yytoken; astnode_set_postfix(" ", $$); };
return: RETURN         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; astnode_set_postfix(" ", $$);};
kernel: KERNEL         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("__global__ void \n#if MAX_THREADS_PER_BLOCK\n__launch_bounds__(MAX_THREADS_PER_BLOCK)\n#endif\n", $$); $$->token = 255 + yytoken; };
sum: SUM               { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("sum", $$); $$->token = 255 + yytoken; };
max: MAX               { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer("max", $$); $$->token = 255 + yytoken; };
hostdefine: HOSTDEFINE {
        $$ = astnode_create(NODE_HOSTDEFINE, NULL, NULL);
        astnode_set_buffer(yytext, $$);
        $$->token = 255 + yytoken;

        astnode_set_prefix("#", $$);

        // Ugly hack
        const char* def_in = "hostdefine";
        const char* def_out = "define";
        assert(strlen(def_in) > strlen(def_out));
        assert(!strncmp($$->buffer, def_in, strlen(def_in)));

        for (size_t i = 0; i < strlen(def_in); ++i)
            $$->buffer[i] = ' ';
        strcpy($$->buffer, def_out);
        $$->buffer[strlen(def_out)] = ' ';

        astnode_set_postfix("\n", $$);
    };

/*
 * =============================================================================
 * Types
 * =============================================================================
*/
type_specifier: int     { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | uint   { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | int3    { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | real    { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | real3   { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | matrix  { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | field   { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              | stencil { $$ = astnode_create(NODE_TSPEC, $1, NULL); }
              ;

type_qualifier: kernel { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | sum    { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              | max    { $$ = astnode_create(NODE_TQUAL, $1, NULL); }
              ;

/*
 * =============================================================================
 * Operators
 * =============================================================================
*/
binary_op: '+'         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
         | '-'         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
         | '/'         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
         | '*'         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
         | '<'         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
         | '>'         { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
         | BINARY_OP   { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
         ;

unary_op: '-'        { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
        | '!'        { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
        ;

assignment_op: ASSIGNOP    { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->token = 255 + yytoken; }
             ;

/*
 * =============================================================================
 * Expressions
 * =============================================================================
*/
primary_expression: identifier         { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                  | number             { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                  | string             { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                  | '(' expression ')' { $$ = astnode_create(NODE_UNKNOWN, $2, NULL); astnode_set_prefix("(", $$); astnode_set_postfix(")", $$); }
                  ;

postfix_expression: primary_expression                         { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                  | postfix_expression '[' expression ']'      { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix("[", $$); astnode_set_postfix("]", $$); }
                  | postfix_expression '(' ')'                 { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); }
                  | postfix_expression '(' expression_list ')' { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); }
                  | postfix_expression '.' identifier          { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix(".", $$); set_identifier_type(NODE_MEMBER_ID, $$->rhs); }
                  | type_specifier '(' expression_list ')'     { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); $$->lhs->type ^= NODE_TSPEC; /* Unset NODE_TSPEC flag, casts are handled as functions */ }
                  ;

declaration_postfix_expression: identifier                                        { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                              | declaration_postfix_expression '[' expression ']' { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix("[", $$); astnode_set_postfix("]", $$); }
                              | declaration_postfix_expression '.' identifier     { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix(".", $$); set_identifier_type(NODE_MEMBER_ID, $$->rhs); }
                              ;

unary_expression: postfix_expression          { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                | unary_op postfix_expression { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                ;

binary_expression: binary_op unary_expression { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                 ;

expression: unary_expression             { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
          | expression binary_expression { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
          ;

expression_list: expression                     { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
               | expression_list ',' expression { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix(",", $$); }
               ;

/*
 * =============================================================================
 * Definitions and Declarations
 * =============================================================================
*/
variable_definition: declaration { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); astnode_set_postfix(";", $$); }
                   | assignment  { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); astnode_set_postfix(";", $$); }
                   ;

declaration: type_declaration declaration_list { $$ = astnode_create(NODE_DECLARATION, $1, $2); }
           ;

declaration_list: declaration_postfix_expression                      { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                | declaration_list ',' declaration_postfix_expression { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix(";", $$); /* Note ';' infix */ }
                ;

parameter: type_declaration identifier { $$ = astnode_create(NODE_DECLARATION, $1, $2); }
         ;

parameter_list: parameter                    { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
              | parameter_list ',' parameter { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix(",", $$); }
              ;

type_declaration: /* Empty */                   { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL);}
                | type_qualifier                { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                | type_specifier                { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                | type_qualifier type_specifier { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                ;

assignment: declaration assignment_body { $$ = astnode_create(NODE_ASSIGNMENT, $1, $2); }
          ;

assignment_body: assignment_op expression_list {
                    $$ = astnode_create(NODE_UNKNOWN, $1, $2);

                    // If more than one expression, it's an array declaration
                    if ($$->rhs && $$->rhs->rhs) {
                        astnode_set_prefix("[]", $$);
                        astnode_set_infix("{", $$);
                        astnode_set_postfix("}", $$);
                    }
                }
               ;

/*
 * =============================================================================
 * Statements
 * =============================================================================
*/
statement: variable_definition  { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
         | selection_statement  { $$ = astnode_create(NODE_BEGIN_SCOPE, $1, NULL); }
         | iteration_statement  { $$ = astnode_create(NODE_BEGIN_SCOPE, $1, NULL); }
         | return expression    { $$ = astnode_create(NODE_UNKNOWN, $1, $2); astnode_set_postfix(";", $$); }
         | function_call        { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); astnode_set_postfix(";", $$); }
         ;

statement_list: statement                { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
              | statement_list statement { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
              ;

compound_statement: '{' '}'                { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_prefix("{", $$); astnode_set_postfix("}", $$); }
                  | '{' statement_list '}' { $$ = astnode_create(NODE_BEGIN_SCOPE, $2, NULL); astnode_set_prefix("{", $$); astnode_set_postfix("}", $$); }
                  ;

selection_statement: if if_statement        { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                   ;

if_statement: expression compound_statement { $$ = astnode_create(NODE_UNKNOWN, $1, $2); astnode_set_prefix("(", $$); astnode_set_infix(")", $$); }
            | expression elif_statement     { $$ = astnode_create(NODE_UNKNOWN, $1, $2); astnode_set_prefix("(", $$); astnode_set_infix(")", $$); }
            | expression else_statement     { $$ = astnode_create(NODE_UNKNOWN, $1, $2); astnode_set_prefix("(", $$); astnode_set_infix(")", $$); }
            ;

elif_statement: compound_statement elif_statement { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
              | elif if_statement                 { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
              ;

else_statement: compound_statement else_statement { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
              | else compound_statement           { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
              ;

iteration_statement: while_statement compound_statement { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                   | for_statement compound_statement   { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                   ;

while_statement: while expression { $$ = astnode_create(NODE_UNKNOWN, $1, $2); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); }
               ;

for_statement: for for_expression { $$ = astnode_create(NODE_UNKNOWN, $1, $2); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); }
             ;

for_expression: identifier range_expression {
    $$ = astnode_create(NODE_UNKNOWN, $1, $2);

    if ($$->rhs->rhs->type & NODE_RANGE) {
        astnode_set_infix("=", $$);

        const size_t padding = 32;
        char* tmp = malloc(strlen($1->buffer) + padding);
        sprintf(tmp, ";%s<", $1->buffer);
        astnode_set_buffer(tmp, $$->rhs->rhs);

        sprintf(tmp, ";++%s", $1->buffer);
        astnode_set_postfix(tmp, $$);
        free(tmp);
    }
};

range_expression: in expression { $$ = astnode_create(NODE_UNKNOWN, NULL, $2); astnode_set_infix(":", $$); } // Note: in keyword skipped
                | in range      { $$ = astnode_create(NODE_UNKNOWN, NULL, $2); }
                ;

range: expression ':' expression { $$ = astnode_create(NODE_RANGE, $1, $3); }
     ;

/*
 * =============================================================================
 * Functions
 * =============================================================================
*/
function_definition: declaration function_body {
                        $$ = astnode_create(NODE_FUNCTION, $1, $2);

                        ASTNode* fn_identifier = get_node_by_token(IDENTIFIER, $$->lhs);
                        assert(fn_identifier);
                        set_identifier_type(NODE_FUNCTION_ID, fn_identifier);

                        const ASTNode* is_kernel = get_node_by_token(KERNEL, $$);
                        if (is_kernel) {
                            $$->type |= NODE_KFUNCTION;
                            set_identifier_type(NODE_KFUNCTION_ID, fn_identifier);

                            // Kernel function parameters
                            astnode_set_prefix("(const int3 start, const int3 end, VertexBufferArray vba", $$->rhs);

                            // Set kernel built-in variables
                            ASTNode* compound_statement = $$->rhs->rhs;
                            assert(compound_statement);

                            astnode_set_prefix("{", compound_statement);
                            astnode_set_postfix(
                              //"\n#pragma unroll\n"
                              //"for (int field = 0; field < NUM_FIELDS; ++field)"
                              //"if (!isnan(out_buffer[field]))"
                              //"vba.out[field][idx] = out_buffer[field];"
                              "}", compound_statement);
                        } else {
                            astnode_set_infix(" __attribute__((unused)) =[&]", $$);
                            astnode_set_postfix(";", $$);
                            $$->type |= NODE_DFUNCTION;
                            //set_identifier_type(NODE_DFUNCTION_ID, fn_identifier);

                            // Pass device function parameters by const reference
                            if ($$->rhs->lhs) {
                                set_identifier_prefix("const ", $$->rhs->lhs);
                                set_identifier_infix("&", $$->rhs->lhs);
                            }

                        }
                    }
                   ;

function_body: '(' ')' compound_statement                { $$ = astnode_create(NODE_BEGIN_SCOPE, NULL, $3); astnode_set_prefix("(", $$); astnode_set_infix(")", $$); }
             | '(' parameter_list ')' compound_statement { $$ = astnode_create(NODE_BEGIN_SCOPE, $2, $4); astnode_set_prefix("(", $$); astnode_set_infix(")", $$); }
             ;

function_call: declaration '(' ')'                 { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); }
             | declaration '(' expression_list ')' { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix("(", $$); astnode_set_postfix(")", $$); }
             ;

/*
 * =============================================================================
 * Stencils
 * =============================================================================
*/
assignment_body_designated: assignment_op expression { $$ = astnode_create(NODE_UNKNOWN, $1, $2); astnode_set_infix("\"", $$); astnode_set_postfix("\"", $$); }
          ;

stencilpoint: stencil_index_list assignment_body_designated { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
            ;

stencil_index: '[' expression ']' { $$ = astnode_create(NODE_UNKNOWN, $2, NULL); astnode_set_prefix("[STENCIL_ORDER/2 +", $$); astnode_set_postfix("]", $$); }
     ;

stencil_index_list: stencil_index            { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
          | stencil_index_list stencil_index { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
          ;

stencilpoint_list: stencilpoint                       { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                 | stencilpoint_list ',' stencilpoint { $$ = astnode_create(NODE_UNKNOWN, $1, $3); astnode_set_infix(",", $$); }
                 ;

stencil_body: '{' stencilpoint_list '}' { $$ = astnode_create(NODE_UNKNOWN, $2, NULL); astnode_set_prefix("{", $$); astnode_set_postfix("},", $$); }
            ;

stencil_definition: declaration stencil_body { $$ = astnode_create(NODE_STENCIL, $1, $2); set_identifier_type(NODE_STENCIL_ID, $$->lhs); }
                  ;
%%

void
print(void)
{
    printf("%s\n", yytext);
}

int
yyerror(const char* str)
{
    fprintf(stderr, "%s on line %d when processing char %d: [%s]\n", str, yyget_lineno(), *yytext, yytext);
    return EXIT_FAILURE;
}

void
set_identifier_type(const NodeType type, ASTNode* curr)
{
    assert(curr);
    if (curr->token == IDENTIFIER) {
        curr->type |= type;
        return;
    }

    if (curr->rhs)
        set_identifier_type(type, curr->rhs);
    if (curr->lhs)
        set_identifier_type(type, curr->lhs);
}

void
set_identifier_prefix(const char* prefix, ASTNode* curr)
{
    assert(curr);
    if (curr->token == IDENTIFIER) {
        astnode_set_prefix(prefix, curr);
        return;
    }

    if (curr->rhs)
      set_identifier_prefix(prefix, curr->rhs);
    if (curr->lhs)
      set_identifier_prefix(prefix, curr->lhs);
}

void
set_identifier_infix(const char* infix, ASTNode* curr)
{
    assert(curr);
    if (curr->token == IDENTIFIER) {
        astnode_set_infix(infix, curr);
        return;
    }

    if (curr->rhs)
      set_identifier_infix(infix, curr->rhs);
    if (curr->lhs)
      set_identifier_infix(infix, curr->lhs);
}

ASTNode*
get_node(const NodeType type, ASTNode* node)
{
  assert(node);

  if (node->type & type)
    return node;
  else if (node->lhs && get_node(type, node->lhs))
    return get_node(type, node->lhs);
  else if (node->rhs && get_node(type, node->rhs))
    return get_node(type, node->rhs);
  else
    return NULL;
}

ASTNode*
get_node_by_token(const int token, ASTNode* node)
{
  assert(node);

  if (node->token == token)
    return node;
  else if (node->lhs && get_node_by_token(token, node->lhs))
    return get_node_by_token(token, node->lhs);
  else if (node->rhs && get_node_by_token(token, node->rhs))
    return get_node_by_token(token, node->rhs);
  else
    return NULL;
}
