%{
#include <stdio.h>
#include <string.h>

#include "ast.h"

extern char* yytext;

int yylex();
int yyerror(const char* str);
int yyget_lineno();

#define YYSTYPE ASTNode* // Sets the default type
%}

%token CONSTANT IN OUT UNIFORM
%token IDENTIFIER NUMBER REAL_NUMBER DOUBLE_NUMBER
%token RETURN
%token SCALAR VECTOR MATRIX SCALARFIELD SCALARARRAY
%token VOID INT INT3 COMPLEX
%token IF ELSE FOR WHILE ELIF
%token LAND LOR BINARY_OP
%token KERNEL DEVICE PREPROCESSED
%token INPLACE_INC INPLACE_DEC

%%

root: program { root->lhs = $1; }
    ;

program: /* Empty*/                                                                     { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); }
       | program function_definition                                                    { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
       | program assignment ';'        /* Global definition */                          { $$ = astnode_create(NODE_UNKNOWN, $1, $2); $$->postfix = ';'; }
       | program declaration ';'       /* Global declaration */                         { $$ = astnode_create(NODE_UNKNOWN, $1, $2); $$->postfix = ';'; }
       ;

/*
 * =============================================================================
 * Functions
 * =============================================================================
 */

function_definition: function_declaration compound_statement                            { $$ = astnode_create(NODE_FUNCTION_DEFINITION, $1, $2); }
                   ;

function_declaration: declaration function_parameter_declaration                        { $$ = astnode_create(NODE_FUNCTION_DECLARATION, $1, $2); }
                    ;

function_parameter_declaration: '(' ')'                                                 { $$ = astnode_create(NODE_FUNCTION_PARAMETER_DECLARATION, NULL, NULL);  $$->prefix = '('; $$->postfix = ')'; }
                              | '(' declaration_list ')'                                { $$ = astnode_create(NODE_FUNCTION_PARAMETER_DECLARATION, $2, NULL);    $$->prefix = '('; $$->postfix = ')'; }
                              ;

/*
 * =============================================================================
 * Statement
 * =============================================================================
 */
statement_list: statement                                                               { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
              | statement_list statement                                                { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
              ;

compound_statement: '{' '}'                                                             { $$ = astnode_create(NODE_COMPOUND_STATEMENT, NULL, NULL); $$->prefix = '{'; $$->postfix = '}'; }
                  | '{' statement_list '}'                                              { $$ = astnode_create(NODE_COMPOUND_STATEMENT, $2, NULL);   $$->prefix = '{'; $$->postfix = '}'; }
                  ;

statement: selection_statement                                                          { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
         | iteration_statement                                                          { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
         | exec_statement ';'                                                           { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); $$->postfix = ';'; }
         //| compound_statement                                                           { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); } // shift-reduce conflict, would be needed only for nested compound statements. TODO remove
         ;

selection_statement: IF expression else_selection_statement                             { $$ = astnode_create(NODE_UNKNOWN, $2, $3); $$->prefix = IF; }
                   ;

else_selection_statement: compound_statement                                            { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                        | compound_statement elif_selection_statement                   { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                        | compound_statement ELSE compound_statement                    { $$ = astnode_create(NODE_UNKNOWN, $1, $3); $$->infix = ELSE; }
                        ;

elif_selection_statement: ELIF expression else_selection_statement                      { $$ = astnode_create(NODE_UNKNOWN, $2, $3); $$->prefix = ELIF; }
                        ;

iteration_statement: WHILE expression compound_statement                                { $$ = astnode_create(NODE_ITERATION_STATEMENT, $2, $3); $$->prefix = WHILE; }
                   | FOR for_expression compound_statement                              { $$ = astnode_create(NODE_ITERATION_STATEMENT, $2, $3); $$->prefix = FOR; }
                   ;

for_expression: '(' for_init_param for_other_params ')'                                 { $$ = astnode_create(NODE_FOR_EXPRESSION, $2, $3); $$->prefix = '('; $$->postfix = ')'; }
              ;

for_init_param: expression ';'                                                          { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); $$->postfix = ';'; }
              | assignment ';'                                                          { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); $$->postfix = ';'; }
              ;

for_other_params: expression ';'                                                        { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); $$->postfix = ';'; }
                | expression ';' expression                                             { $$ = astnode_create(NODE_UNKNOWN, $1, $3); $$->infix = ';'; }
                ;

exec_statement: declaration                                                             { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
              | assignment                                                              { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
              | expression                                                              { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
              | return return_statement                                                 { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
              ;

assignment: declaration '=' expression                                                  { $$ = astnode_create(NODE_UNKNOWN, $1, $3); $$->infix = '='; }
          | declaration '(' expression_list ')'                                         { $$ = astnode_create(NODE_UNKNOWN, $1, $3); $$->infix = '('; $$->postfix = ')'; } // C++ style initializer
          | expression '=' expression                                                   { $$ = astnode_create(NODE_UNKNOWN, $1, $3); $$->infix = '='; }
          ;

return_statement: /* Empty */                                                           { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); }
                | expression                                                            { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                ;

/*
 * =============================================================================
 * Declaration
 * =============================================================================
 */

declaration_list: declaration                                                           { $$ = astnode_create(NODE_DECLARATION_LIST, $1, NULL); }
                | declaration_list ',' declaration                                      { $$ = astnode_create(NODE_DECLARATION_LIST, $1, $3); $$->infix = ','; }
                ;

declaration: type_declaration identifier                                                { $$ = astnode_create(NODE_DECLARATION, $1, $2); } // Note: accepts only one type qualifier. Good or not?
           | type_declaration array_declaration                                         { $$ = astnode_create(NODE_DECLARATION, $1, $2); }
           ;

array_declaration: identifier '[' ']'                                                   { $$ = astnode_create(NODE_ARRAY_DECLARATION, $1, NULL); $$->infix = '['; $$->postfix = ']'; }
                 | identifier '[' expression ']'                                        { $$ = astnode_create(NODE_ARRAY_DECLARATION, $1, $3); $$->infix = '['; $$->postfix = ']'; }
                 ;

type_declaration: type_specifier                                                        { $$ = astnode_create(NODE_TYPE_DECLARATION, $1, NULL); }
                | type_qualifier type_specifier                                         { $$ = astnode_create(NODE_TYPE_DECLARATION, $1, $2); }
                ;

/*
 * =============================================================================
 * Expressions
 * =============================================================================
 */
expression_list: expression                                                             { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
               | expression_list ',' expression                                         { $$ = astnode_create(NODE_UNKNOWN, $1, $3); $$->infix = ','; }
               ;

expression: unary_expression                                                            { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
          | expression binary_expression                                                { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
          ;

binary_expression: binary_operator unary_expression                                     { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                 ;

unary_expression: postfix_expression                                                    { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                | unary_operator postfix_expression                                     { $$ = astnode_create(NODE_UNKNOWN, $1, $2); }
                ;

postfix_expression: primary_expression                                                  { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                  | postfix_expression '[' expression_list ']' /* Subscript */          { $$ = astnode_create(NODE_MULTIDIM_SUBSCRIPT_EXPRESSION, $1, $3);    $$->infix = '['; $$->postfix = ']'; }
                  | cast_expression '{' expression_list '}'    /* Array */              { $$ = astnode_create(NODE_UNKNOWN, $1, $3); $$->infix = '{'; $$->postfix = '}'; }
                  | postfix_expression '(' ')'                 /* Function call */      { $$ = astnode_create(NODE_UNKNOWN, $1, NULL);  $$->infix = '('; $$->postfix = ')'; }
                  | postfix_expression '(' expression_list ')' /* Function call */      { $$ = astnode_create(NODE_UNKNOWN, $1, $3);    $$->infix = '('; $$->postfix = ')'; }
                  | type_specifier '(' expression_list ')'     /* Cast */               { $$ = astnode_create(NODE_UNKNOWN, $1, $3);  $$->infix = '('; $$->postfix = ')'; }
                  | postfix_expression '.' identifier          /* Member access */      { $$ = astnode_create(NODE_UNKNOWN, $1, $3);    $$->infix = '.'; }
                  ;

cast_expression: /* Empty: implicit cast */                                             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); }
               | '(' type_specifier ')'                                                 { $$ = astnode_create(NODE_UNKNOWN, $2, NULL); $$->prefix = '('; $$->postfix = ')'; }
               ;

primary_expression: identifier                                                          { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                  | number                                                              { $$ = astnode_create(NODE_UNKNOWN, $1, NULL); }
                  | '(' expression ')'                                                  { $$ = astnode_create(NODE_UNKNOWN, $2, NULL); $$->prefix = '('; $$->postfix = ')'; }
                  ;



/*
 * =============================================================================
 * Terminals
 * =============================================================================
 */

binary_operator: '+'                                                                    { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); $$->infix = yytext[0]; }
               | '-'                                                                    { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); $$->infix = yytext[0]; }
               | '/'                                                                    { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); $$->infix = yytext[0]; }
               | '*'                                                                    { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); $$->infix = yytext[0]; }
               | '<'                                                                    { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); $$->infix = yytext[0]; }
               | '>'                                                                    { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); $$->infix = yytext[0]; }
               | LAND                                                                   { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); }
               | LOR                                                                    { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); }
               | BINARY_OP                                                              { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); }
               ;

unary_operator: '-' /* C-style casts are disallowed, would otherwise be defined here */ { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); $$->infix = yytext[0]; }
              | '!'                                                                     { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); $$->infix = yytext[0]; }
              | INPLACE_INC                                                             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); $$->token = INPLACE_INC; }
              | INPLACE_DEC                                                             { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); $$->token = INPLACE_DEC; }
              ;

type_qualifier: KERNEL                                                                  { $$ = astnode_create(NODE_TYPE_QUALIFIER, NULL, NULL); $$->token = KERNEL; }
              | DEVICE                                                                  { $$ = astnode_create(NODE_TYPE_QUALIFIER, NULL, NULL); $$->token = DEVICE; }
              | PREPROCESSED                                                            { $$ = astnode_create(NODE_TYPE_QUALIFIER, NULL, NULL); $$->token = PREPROCESSED; }
              | CONSTANT                                                                { $$ = astnode_create(NODE_TYPE_QUALIFIER, NULL, NULL); $$->token = CONSTANT; }
              | IN                                                                      { $$ = astnode_create(NODE_TYPE_QUALIFIER, NULL, NULL); $$->token = IN; }
              | OUT                                                                     { $$ = astnode_create(NODE_TYPE_QUALIFIER, NULL, NULL); $$->token = OUT; }
              | UNIFORM                                                                 { $$ = astnode_create(NODE_TYPE_QUALIFIER, NULL, NULL); $$->token = UNIFORM; }
              ;

type_specifier: VOID                                                                    { $$ = astnode_create(NODE_TYPE_SPECIFIER, NULL, NULL); $$->token = VOID; }
              | INT                                                                     { $$ = astnode_create(NODE_TYPE_SPECIFIER, NULL, NULL); $$->token = INT; }
              | INT3                                                                    { $$ = astnode_create(NODE_TYPE_SPECIFIER, NULL, NULL); $$->token = INT3; }
              | SCALAR                                                                  { $$ = astnode_create(NODE_TYPE_SPECIFIER, NULL, NULL); $$->token = SCALAR;  }
              | VECTOR                                                                  { $$ = astnode_create(NODE_TYPE_SPECIFIER, NULL, NULL); $$->token = VECTOR;  }
              | MATRIX                                                                  { $$ = astnode_create(NODE_TYPE_SPECIFIER, NULL, NULL); $$->token = MATRIX;  }
              | SCALARFIELD                                                             { $$ = astnode_create(NODE_TYPE_SPECIFIER, NULL, NULL); $$->token = SCALARFIELD;  }
              | SCALARARRAY                                                             { $$ = astnode_create(NODE_TYPE_SPECIFIER, NULL, NULL); $$->token = SCALARARRAY;  }
              | COMPLEX                                                                 { $$ = astnode_create(NODE_TYPE_SPECIFIER, NULL, NULL); $$->token = COMPLEX;  }
              ;

identifier: IDENTIFIER                                                                  { $$ = astnode_create(NODE_IDENTIFIER, NULL, NULL); astnode_set_buffer(yytext, $$); }
          ;

number: REAL_NUMBER                                                                     { $$ = astnode_create(NODE_REAL_NUMBER, NULL, NULL); astnode_set_buffer(yytext, $$); }
      | DOUBLE_NUMBER                                                                   { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); $$->buffer[strlen($$->buffer) - 1] = '\0'; }
      | NUMBER                                                                          { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); }
      ;

return: RETURN                                                                          { $$ = astnode_create(NODE_UNKNOWN, NULL, NULL); astnode_set_buffer(yytext, $$); }
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
