/*
    Copyright (C) 2014-2021, Johannes Pekkila, Miikka Vaisala.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * @file
 * \brief Brief info.
 *
 * Detailed info.
 *
 */

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "acc.tab.h"
#include "ast.h"

ASTNode* root = NULL;

// Output files
static FILE* DSLHEADER  = NULL;
static FILE* CUDAHEADER = NULL;
static FILE* KHEADER    = NULL;
static FILE* DHEADER    = NULL;
static FILE* FHEADER    = NULL;

static const char* dslheader_filename  = "user_defines.h";
static const char* cudaheader_filename = "user_kernels.cuh";
static const char* kheader_filename    = "user_kernel_decl.h";
static const char* dheader_filename    = "user_kernels.h";
static const char* fheader_filename    = "astaroth.f90";

// Forward declaration of yyparse
int yyparse(void);

/*
 * =============================================================================
 * Translation
 * =============================================================================
 */
#define TRANSLATION_TABLE_SIZE (1024)
static const char* translation_table[TRANSLATION_TABLE_SIZE] = {
    [0] = NULL,
    // Control flow
    [IF]    = "if",
    [ELSE]  = "else",
    [ELIF]  = "else if",
    [WHILE] = "while",
    [FOR]   = "for",
    // Type specifiers
    [VOID]        = "void",
    [INT]         = "int",
    [INT3]        = "int3",
    [SCALAR]      = "AcReal",
    [VECTOR]      = "AcReal3",
    [MATRIX]      = "AcMatrix",
    [SCALARFIELD] = "AcReal",
    [SCALARARRAY] = "const AcReal* __restrict__",
    [COMPLEX]     = "acComplex",
    // Type qualifiers
    [KERNEL]       = "template <int step_number>  static __global__",
    [DEVICE]       = "static __device__ __forceinline__",
    [PREPROCESSED] = "static __device__ __forceinline__",
    [CONSTANT]     = "const",
    [IN]           = "in",
    [OUT]          = "out",
    [UNIFORM]      = "uniform",
    // ETC
    [INPLACE_INC] = "++",
    [INPLACE_DEC] = "--", // TODO remove, astnodesetbuffer
    // Unary
    [','] = ",",
    [';'] = ";\n",
    ['('] = "(",
    [')'] = ")",
    ['['] = "[",
    [']'] = "]",
    ['{'] = "{\n",
    ['}'] = "}\n",
    ['='] = "=",
    ['+'] = "+",
    ['-'] = "-",
    ['/'] = "/",
    ['*'] = "*",
    ['<'] = "<",
    ['>'] = ">",
    ['!'] = "!",
    ['.'] = ".",
};

static const char*
translate(const int token)
{
    assert(token >= 0);
    assert(token < TRANSLATION_TABLE_SIZE);
    if (token > 0) {
        if (!translation_table[token])
            fprintf(stderr, "Error: unidentified token %d\n", token);
        assert(translation_table[token]);
    }

    return translation_table[token];
}

/*
 * =============================================================================
 * Symbols
 * =============================================================================
 */
typedef enum {
    SYMBOLTYPE_FUNCTION,
    SYMBOLTYPE_FUNCTION_PARAMETER,
    SYMBOLTYPE_OTHER,
    NUM_SYMBOLTYPES
} SymbolType;

#define MAX_ID_LEN (256)
typedef struct {
    SymbolType type;
    int type_qualifier;
    int type_specifier;
    char identifier[MAX_ID_LEN];
} Symbol;

#define SYMBOL_TABLE_SIZE (65536)
static Symbol symbol_table[SYMBOL_TABLE_SIZE] = {};

#define MAX_NESTS (32)
static size_t num_symbols[MAX_NESTS] = {};
static size_t current_nest           = 0;

static Symbol*
symboltable_lookup(const char* identifier)
{
    // TODO assert that symbol not function! cannot be since we allow overloads->conflicts if not
    // explicit

    if (!identifier)
        return NULL;

    for (size_t i = 0; i < num_symbols[current_nest]; ++i)
        if (strcmp(identifier, symbol_table[i].identifier) == 0)
            return &symbol_table[i];

    return NULL;
}

static void
add_symbol(const SymbolType type, const int tqualifier, const int tspecifier, const char* id)
{
    assert(num_symbols[current_nest] < SYMBOL_TABLE_SIZE);

    if (symboltable_lookup(id) && type != SYMBOLTYPE_FUNCTION) {
        fprintf(stderr,
                "Syntax error. Symbol '%s' is ambiguous, declared multiple times in the same scope"
                " (shadowing).\n",
                id);
        assert(0);
    }

    symbol_table[num_symbols[current_nest]].type           = type;
    symbol_table[num_symbols[current_nest]].type_qualifier = tqualifier;
    symbol_table[num_symbols[current_nest]].type_specifier = tspecifier;
    strcpy(symbol_table[num_symbols[current_nest]].identifier, id);

    ++num_symbols[current_nest];
}

static void
print_symbol(const Symbol* symbol)
{
    const char* fields[] = {
        translate(symbol->type_qualifier),
        translate(symbol->type_specifier),
        symbol->identifier,
    };

    const size_t num_fields = sizeof(fields) / sizeof(fields[0]);
    for (size_t i = 0; i < num_fields; ++i)
        if (fields[i])
            fprintf(CUDAHEADER, "%s ", fields[i]);
}

static inline void
print_symbol_table(void)
{
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        printf("%lu: ", i);
        const char* fields[] = {translate(symbol_table[i].type_qualifier),
                                translate(symbol_table[i].type_specifier),
                                symbol_table[i].identifier};

        const size_t num_fields = sizeof(fields) / sizeof(fields[0]);
        for (size_t j = 0; j < num_fields; ++j)
            if (fields[j])
                printf("%s ", fields[j]);

        if (symbol_table[i].type == SYMBOLTYPE_FUNCTION)
            printf("(function)");
        else if (symbol_table[i].type == SYMBOLTYPE_FUNCTION_PARAMETER)
            printf("(function parameter)");
        else
            printf("(other)");
        printf("\n");
    }
}

/*
 * =============================================================================
 * Traversal state
 * =============================================================================
 */
static bool inside_declaration = false;
static bool inside_kernel      = false;
static bool inside_function    = false;

/*
 * =============================================================================
 * AST traversal
 * =============================================================================
 */
static void
traverse(const ASTNode* node)
{
    // Prefix translation
    if (!inside_declaration && translate(node->prefix))
        fprintf(CUDAHEADER, "%s", translate(node->prefix));

    // Prefix logic
    if (node->type == NODE_COMPOUND_STATEMENT) {
        assert(current_nest < MAX_NESTS);

        ++current_nest;
        num_symbols[current_nest] = num_symbols[current_nest - 1];
    }
    if (node->type == NODE_DECLARATION)
        inside_declaration = true;
    if (node->type == NODE_FUNCTION_DEFINITION)
        inside_function = true;
    if (node->token == KERNEL)
        inside_kernel = true;

    if (node->type == NODE_FUNCTION_PARAMETER_DECLARATION) {
        // Boilerplates
        const ASTNode* typedecl = node->parent->lhs->lhs;
        const ASTNode* typequal = typedecl->lhs;
        assert(typedecl->type == NODE_TYPE_DECLARATION);
        if (typequal->type == NODE_TYPE_QUALIFIER) {
            if (typequal->token == KERNEL) {
                fprintf(CUDAHEADER, "GEN_KERNEL_PARAM_BOILERPLATE");
                if (node->lhs != NULL) {
                    fprintf(stderr, "Syntax error: function parameters for Kernel functions not "
                                    "allowed!\n");
                }
            }
            else if (typequal->token == PREPROCESSED) {
                fprintf(CUDAHEADER, "GEN_PREPROCESSED_PARAM_BOILERPLATE, ");
            }
        }
    }

    if (node->type == NODE_COMPOUND_STATEMENT) {
        if (node->parent->type == NODE_FUNCTION_DEFINITION) {
            const Symbol* symbol = symboltable_lookup(node->parent->lhs->lhs->rhs->buffer);
            if (symbol && symbol->type_qualifier == KERNEL) {
                fprintf(CUDAHEADER, "GEN_KERNEL_BUILTIN_VARIABLES_BOILERPLATE();");
                for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
                    if (symbol_table[i].type_qualifier == IN) {
                        fprintf(CUDAHEADER, "const %sData %s = READ(handle_%s);\n",
                                translate(symbol_table[i].type_specifier),
                                symbol_table[i].identifier, symbol_table[i].identifier);
                    }
                    else if (symbol_table[i].type_qualifier == OUT) {
                        fprintf(CUDAHEADER, "%s %s = READ_OUT(handle_%s);",
                                translate(symbol_table[i].type_specifier),
                                symbol_table[i].identifier, symbol_table[i].identifier);
                    }
                }
            }
        }
    }

    // Traverse LHS
    if (node->lhs)
        traverse(node->lhs);

    // Infix translation
    if (!inside_declaration && translate(node->infix))
        fprintf(CUDAHEADER, "%s", translate(node->infix));

    // Infix logic
    // If the node is a subscript expression and the expression list inside it is not empty
    if (node->type == NODE_MULTIDIM_SUBSCRIPT_EXPRESSION && node->rhs)
        fprintf(CUDAHEADER, "IDX(");

    // Traverse RHS
    if (node->rhs)
        traverse(node->rhs);

    // Add new symbols to the symbol table
    if (node->type == NODE_DECLARATION) {
        int stype;
        ASTNode* tmp = node->parent;
        while (tmp->type == NODE_DECLARATION_LIST)
            tmp = tmp->parent;

        if (tmp->type == NODE_FUNCTION_DECLARATION)
            stype = SYMBOLTYPE_FUNCTION;
        else if (tmp->type == NODE_FUNCTION_PARAMETER_DECLARATION)
            stype = SYMBOLTYPE_FUNCTION_PARAMETER;
        else if (node->parent && node->parent->parent && node->parent->parent->parent &&
                 node->parent->parent->parent->type == NODE_FOR_EXPRESSION)
            stype = SYMBOLTYPE_FUNCTION_PARAMETER;
        else
            stype = SYMBOLTYPE_OTHER;

        const ASTNode* tdeclaration = node->lhs;
        const int tqualifier        = tdeclaration->rhs ? tdeclaration->lhs->token : 0;
        const int tspecifier        = tdeclaration->rhs ? tdeclaration->rhs->token
                                                        : tdeclaration->lhs->token;

        const char* identifier = node->rhs->type == NODE_IDENTIFIER ? node->rhs->buffer
                                                                    : node->rhs->lhs->buffer;
        add_symbol(stype, tqualifier, tspecifier, identifier);

        // Translate the new symbol
        if (tqualifier == UNIFORM) {
            if (tspecifier == SCALAR || tspecifier == VECTOR || tspecifier == INT ||
                tspecifier == INT3) {
                fprintf(CUDAHEADER, "static %s %s_DEFAULT_VALUE", translate(tspecifier),
                        identifier);
            }
            // else do nothing
        }
        else if (tqualifier == KERNEL) {
            fprintf(CUDAHEADER, "%s %s\n%s", //
                    translate(tqualifier), translate(tspecifier), identifier);
        }
        else if (tqualifier == DEVICE) {
            fprintf(CUDAHEADER, "%s %s\n%s", //
                    translate(tqualifier), translate(tspecifier), identifier);
        }
        else if (tqualifier == PREPROCESSED) {
            fprintf(CUDAHEADER, "%s %s\npreprocessed_%s", //
                    translate(tqualifier), translate(tspecifier), identifier);
        }
        else if (stype == SYMBOLTYPE_FUNCTION) {
            // Stencil assembly stage device function
            fprintf(CUDAHEADER, "%s %s\n%s", //
                    translate(DEVICE), translate(tspecifier), identifier);
        }
        else if (stype == SYMBOLTYPE_FUNCTION_PARAMETER) {
            tmp = tmp->parent;
            assert(tmp->type == NODE_FUNCTION_DECLARATION);

            // TODO FIX not to use symboltable_lookup
            const Symbol* parent_function = symboltable_lookup(tmp->lhs->rhs->buffer);
            if (parent_function && (tqualifier == IN || tqualifier == OUT)) {
                if (tmp->lhs->lhs->lhs->token == DEVICE) {
                    fprintf(CUDAHEADER, "const %sData& %s", //
                            translate(tspecifier), identifier);
                }
                else {
                    fprintf(CUDAHEADER, "const __restrict__ %s* %s", //
                            translate(tspecifier), identifier);
                }
            }
            else {
                print_symbol(&symbol_table[num_symbols[current_nest] - 1]);
            }
        }
        else if (tqualifier == IN || tqualifier == OUT) { // Global in/out declarator
            fprintf(CUDAHEADER, "static __device__ const ");
            fprintf(CUDAHEADER, "%s ", tspecifier == SCALARFIELD ? "int" : "int3");
            fprintf(CUDAHEADER, "handle_%s ", identifier);
            fprintf(CUDAHEADER, "%s ", tspecifier == SCALARFIELD ? "" : "= make_int3");
        }
        else {
            // Do a regular translation
            print_symbol(&symbol_table[num_symbols[current_nest] - 1]);
        }

        if (node->rhs->type == NODE_ARRAY_DECLARATION) {
            // Traverse the expression once again, this time with
            // "inside_declaration" flag off
            inside_declaration = false;
            fprintf(CUDAHEADER, "%s ", translate(node->rhs->infix));
            if (node->rhs->rhs)
                traverse(node->rhs->rhs);
            fprintf(CUDAHEADER, "%s ", translate(node->rhs->postfix));
        }
    }
    else {
        // Translate existing symbols
        const Symbol* symbol = symboltable_lookup(node->buffer);

        if (symbol) {
            // Uniforms
            if (symbol->type_qualifier == UNIFORM) {
                if (inside_kernel && symbol->type_specifier == SCALARARRAY)
                    fprintf(CUDAHEADER, "buffer.profiles[%s] ", symbol->identifier);
                else if (!inside_function &&
                         ((symbol->type_specifier == SCALAR) ||
                          (symbol->type_specifier == VECTOR) || (symbol->type_specifier == INT) ||
                          (symbol->type_specifier == INT3))) // Global scope and an uniform which
                                                             // can be set to a default value
                    fprintf(CUDAHEADER, "%s_DEFAULT_VALUE ", symbol->identifier);
                else // Use device constants inside device functions
                    fprintf(CUDAHEADER, "DCONST(%s) ", symbol->identifier);
            }
            else if (!inside_declaration) {
                // Regular symbol translation
                if (translate(node->token))
                    fprintf(CUDAHEADER, "%s ", translate(node->token));
                if (node->buffer)
                    fprintf(CUDAHEADER, "%s ", node->buffer);
            }
        }
        else if (!inside_declaration) {
            // Literal translation
            if (translate(node->token))
                fprintf(CUDAHEADER, "%s ", translate(node->token));
            if (node->buffer) {
                if (node->type == NODE_REAL_NUMBER) {
                    fprintf(CUDAHEADER, "%s(%s) ", translate(SCALAR),
                            node->buffer); // Cast to correct precision
                }
                else {
                    fprintf(CUDAHEADER, "%s ", node->buffer);
                }
            }
        }
    }

    // Postfix logic
    // If the node is a subscript expression and the expression list inside it is not empty
    if (node->type == NODE_MULTIDIM_SUBSCRIPT_EXPRESSION && node->rhs)
        fprintf(CUDAHEADER, ")"); // Closing bracket of IDX()

    if (node->type == NODE_COMPOUND_STATEMENT) {
        // if (node->type == NODE_FUNCTION_DEFINITION || node->type == NODE_ITERATION_STATEMENT) {
        assert(current_nest > 0);
        --current_nest;

        // Drop function parameters (incl. those declared in for statements)
        while (symbol_table[num_symbols[current_nest] - 1].type == SYMBOLTYPE_FUNCTION_PARAMETER)
            --num_symbols[current_nest];

        inside_kernel = false;

        // Kernel writeback boilerplate
        if (node->parent->type == NODE_FUNCTION_DEFINITION) {
            const Symbol* symbol = symboltable_lookup(node->parent->lhs->lhs->rhs->buffer);
            if (symbol && symbol->type_qualifier == KERNEL) {
                for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
                    if (symbol_table[i].type_qualifier == OUT) {
                        fprintf(CUDAHEADER, "WRITE_OUT(handle_%s, %s);\n",
                                symbol_table[i].identifier, symbol_table[i].identifier);
                    }
                }
            }
        }
    }
    if (node->type == NODE_DECLARATION)
        inside_declaration = false;
    if (node->type == NODE_FUNCTION_DEFINITION)
        inside_function = false;

    // Postfix translation
    if (!inside_declaration && translate(node->postfix))
        fprintf(CUDAHEADER, "%s", translate(node->postfix));
}

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))

static const char* builtin_int_params[] = {
    "AC_nx",
    "AC_ny",
    "AC_nz",
    "AC_mx",
    "AC_my",
    "AC_mz",
    "AC_nx_min",
    "AC_ny_min",
    "AC_nz_min",
    "AC_nx_max",
    "AC_ny_max",
    "AC_nz_max",
    "AC_mxy",
    "AC_nxy",
    "AC_nxyz",
    "AC_bc_type_bot_x",
    "AC_bc_type_bot_y",
    "AC_bc_type_bot_z",
    "AC_bc_type_top_x",
    "AC_bc_type_top_y",
    "AC_bc_type_top_z",

};

static const char* builtin_int3_params[] = {
    "AC_global_grid_n",
    "AC_multigpu_offset",
};

static const char* builtin_real_params[] = {
    "AC_dt",
};

/*
// Add built-in real3 params here and uncomment the other 2x code blocks where
// builtin_real3_params is referenced
static const char* builtin_real3_params[] = {};
*/

static void
generate_preprocessed_structures(void)
{
    // Data structure
    fprintf(CUDAHEADER, "\n");

    // Read data to the data struct
    fprintf(CUDAHEADER, "static __device__ __forceinline__ AcRealData\
            read_data(const int3& vertexIdx,\
                const int3& globalVertexIdx,\
            AcReal* __restrict__ buf[], const int handle)\
            {\n\
                %sData data;\n",
            translate(SCALAR));

    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        if (symbol_table[i].type_qualifier == PREPROCESSED)
            fprintf(CUDAHEADER,
                    "data.%s = preprocessed_%s(vertexIdx, globalVertexIdx, buf[handle]);\n",
                    symbol_table[i].identifier, symbol_table[i].identifier);
    }
    fprintf(CUDAHEADER, "return data;\n");
    fprintf(CUDAHEADER, "}\n");

    // Functions for accessing the data struct members
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        if (symbol_table[i].type_qualifier == PREPROCESSED)
            fprintf(CUDAHEADER, "static __device__ __forceinline__ %s\
                    %s(const AcRealData& data)\
                    {\n\
                        return data.%s;\
                    }\n",
                    translate(symbol_table[i].type_specifier), symbol_table[i].identifier,
                    symbol_table[i].identifier);
    }

    // Syntactic sugar: Vector data struct
    fprintf(CUDAHEADER, "static __device__ __forceinline__ AcReal3Data\
        read_data(const int3& vertexIdx,\
                  const int3& globalVertexIdx,\
                  AcReal* __restrict__ buf[], const int3& handle)\
        {\
            AcReal3Data data;\
        \
            data.x = read_data(vertexIdx, globalVertexIdx, buf, handle.x);\
            data.y = read_data(vertexIdx, globalVertexIdx, buf, handle.y);\
            data.z = read_data(vertexIdx, globalVertexIdx, buf, handle.z);\
        \
            return data;\
        }\
    ");

    const size_t max_buflen = 65536;
    char buffer[max_buflen];
    rewind(CUDAHEADER);
    const size_t buflen = fread(buffer, sizeof(char), max_buflen, CUDAHEADER);
    fclose(CUDAHEADER);
    CUDAHEADER = fopen(cudaheader_filename, "w+");

    fprintf(CUDAHEADER, "#pragma once\n");

    // Add built-in params (the best way would be to inject these to user src with AC syntax)
    for (size_t i = 0; i < ARRAY_SIZE(builtin_int_params); ++i)
        fprintf(CUDAHEADER, "static const int %s_DEFAULT_VALUE = 0;\n", builtin_int_params[i]);

    for (size_t i = 0; i < ARRAY_SIZE(builtin_int3_params); ++i)
        fprintf(CUDAHEADER, "static const int3 %s_DEFAULT_VALUE = make_int3(0, 0, 0);\n",
                builtin_int3_params[i]);

    for (size_t i = 0; i < ARRAY_SIZE(builtin_real_params); ++i)
        fprintf(CUDAHEADER, "static const AcReal %s_DEFAULT_VALUE = (AcReal)0.0;\n",
                builtin_real_params[i]);

    /*
    for (size_t i = 0; i < ARRAY_SIZE(builtin_real3_params); ++i)
        fprintf(CUDAHEADER,
                "static const AcReal3 %s_DEFAULT_VALUE = (AcReal3){(AcReal)0.0, (AcReal)0.0, "
                "(AcReal)0.0};\n",
                builtin_real3_params[i]);
                */

    fprintf(CUDAHEADER, "typedef struct {\n");
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        if (symbol_table[i].type_qualifier == PREPROCESSED)
            fprintf(CUDAHEADER, "%s %s;\n", translate(symbol_table[i].type_specifier),
                    symbol_table[i].identifier);
    }
    fprintf(CUDAHEADER, "} %sData;\n", translate(SCALAR));
    fprintf(CUDAHEADER, "typedef struct {\
                            AcRealData x;\
                            AcRealData y;\
                            AcRealData z;\
                        } AcReal3Data;\n");
    fprintf(CUDAHEADER, "static __device__ AcRealData\
                         read_data(const int3& vertexIdx,\
                                   const int3& globalVertexIdx,\
                                   AcReal* __restrict__ buf[], const int handle);\n");
    fprintf(CUDAHEADER, "static __device__ AcReal3Data\
                         read_data(const int3& vertexIdx,\
                                   const int3& globalVertexIdx,\
                                   AcReal* __restrict__ buf[], const int3& handle);\n");
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        if (symbol_table[i].type_qualifier == PREPROCESSED)
            fprintf(CUDAHEADER, "static __device__ %s %s(const AcRealData& data);\n",
                    translate(symbol_table[i].type_specifier), symbol_table[i].identifier);
    }

    fwrite(buffer, sizeof(char), buflen, CUDAHEADER);
}

static void
generate_headers(void)
{
    // Fortran interface
    const char* fortran_interface = R"(
!  -*-f90-*-  (for emacs)    vim:set filetype=fortran:  (for vim)

! Utils (see astaroth_fortran.cc for definitions)
external achostupdatebuiltinparams
external acgetdevicecount

! Device interface (see astaroth_fortran.cc for definitions)
external acdevicecreate, acdevicedestroy
external acdeviceprintinfo
external acdeviceloadmeshinfo
external acdeviceloadmesh, acdevicestoremesh
external acdeviceintegratesubstep
external acdeviceperiodicboundconds
external acdeviceswapbuffers
external acdevicereducescal, acdevicereducevec
external acdevicesynchronizestream
  )";
    fprintf(FHEADER, "%s\n", fortran_interface);

    fprintf(DSLHEADER, "#pragma once\n");

    // Int params
    fprintf(DSLHEADER, "#define AC_FOR_USER_INT_PARAM_TYPES(FUNC)");
    int enumcounter = 0;
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        if (symbol_table[i].type_specifier == INT && symbol_table[i].type_qualifier == UNIFORM) {
            fprintf(DSLHEADER, "\\\nFUNC(%s)", symbol_table[i].identifier);
            fprintf(FHEADER, "integer(c_int), parameter :: %s = %d\n", symbol_table[i].identifier,
                    enumcounter);
            ++enumcounter;
        }
    }
    fprintf(DSLHEADER, "\n\n");
    fprintf(FHEADER, "integer(c_int), parameter :: AC_NUM_INT_PARAMS = %d\n\n", enumcounter);

    // Int3 params
    fprintf(DSLHEADER, "#define AC_FOR_USER_INT3_PARAM_TYPES(FUNC)");
    enumcounter = 0;
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        if (symbol_table[i].type_specifier == INT3 && symbol_table[i].type_qualifier == UNIFORM) {
            fprintf(DSLHEADER, "\\\nFUNC(%s)", symbol_table[i].identifier);

            fprintf(FHEADER, "integer(c_int), parameter :: %s = %d\n", symbol_table[i].identifier,
                    enumcounter);
            ++enumcounter;
        }
    }
    fprintf(DSLHEADER, "\n\n");
    fprintf(FHEADER, "integer(c_int), parameter :: AC_NUM_INT3_PARAMS = %d\n\n", enumcounter);

    // Scalar params
    fprintf(DSLHEADER, "#define AC_FOR_USER_REAL_PARAM_TYPES(FUNC)");
    enumcounter = 0;
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        if (symbol_table[i].type_specifier == SCALAR && symbol_table[i].type_qualifier == UNIFORM) {
            fprintf(DSLHEADER, "\\\nFUNC(%s)", symbol_table[i].identifier);

            fprintf(FHEADER, "integer(c_int), parameter :: %s = %d\n", symbol_table[i].identifier,
                    enumcounter);
            ++enumcounter;
        }
    }
    fprintf(DSLHEADER, "\n\n");
    fprintf(FHEADER, "integer(c_int), parameter :: AC_NUM_REAL_PARAMS = %d\n\n", enumcounter);

    // Vector params
    fprintf(DSLHEADER, "#define AC_FOR_USER_REAL3_PARAM_TYPES(FUNC)");
    enumcounter = 0;
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        if (symbol_table[i].type_specifier == VECTOR && symbol_table[i].type_qualifier == UNIFORM) {
            fprintf(DSLHEADER, "\\\nFUNC(%s)", symbol_table[i].identifier);

            fprintf(FHEADER, "integer(c_int), parameter :: %s = %d\n", symbol_table[i].identifier,
                    enumcounter);
            ++enumcounter;
        }
    }
    fprintf(DSLHEADER, "\n\n");
    fprintf(FHEADER, "integer(c_int), parameter :: AC_NUM_REAL3_PARAMS = %d\n\n", enumcounter);

    // Scalar fields
    fprintf(DSLHEADER, "#define AC_FOR_VTXBUF_HANDLES(FUNC)");
    enumcounter = 0;
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        if (symbol_table[i].type_specifier == SCALARFIELD &&
            symbol_table[i].type_qualifier == UNIFORM) {
            fprintf(DSLHEADER, "\\\nFUNC(%s)", symbol_table[i].identifier);

            fprintf(FHEADER, "integer(c_int), parameter :: %s = %d\n", symbol_table[i].identifier,
                    enumcounter);
            ++enumcounter;
        }
    }
    fprintf(DSLHEADER, "\n\n");
    fprintf(FHEADER, "integer(c_int), parameter :: AC_NUM_VTXBUF_HANDLES = %d\n\n", enumcounter);

    // Scalar arrays
    fprintf(DSLHEADER, "#define AC_FOR_SCALARARRAY_HANDLES(FUNC)");
    enumcounter = 0;
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        if (symbol_table[i].type_specifier == SCALARARRAY &&
            symbol_table[i].type_qualifier == UNIFORM) {
            fprintf(DSLHEADER, "\\\nFUNC(%s)", symbol_table[i].identifier);

            fprintf(FHEADER, "integer(c_int), parameter :: %s = %d\n", symbol_table[i].identifier,
                    enumcounter);
            ++enumcounter;
        }
    }
    fprintf(DSLHEADER, "\n\n");
    fprintf(FHEADER, "integer(c_int), parameter :: AC_NUM_SCALARRAY_HANDLES = %d\n\n", enumcounter);

    // Streams
    const size_t nstreams = 32;
    for (size_t i = 0; i < nstreams; ++i) {
        fprintf(DSLHEADER, "#define STREAM_%lu (%lu)\n", i, i);
        fprintf(FHEADER, "integer(c_int), parameter :: STREAM_%lu = %lu\n", i, i);
    }
    fprintf(DSLHEADER, "#define NUM_STREAMS (%lu)\n", nstreams);
    fprintf(DSLHEADER, "#define STREAM_DEFAULT (STREAM_0)\n");
    fprintf(DSLHEADER, "#define STREAM_ALL (NUM_STREAMS)\n");
    fprintf(FHEADER, "integer(c_int), parameter :: NUM_STREAMS = %lu\n", nstreams);
    fprintf(FHEADER, "integer(c_int), parameter :: STREAM_DEFAULT = STREAM_0\n");
    fprintf(FHEADER, "integer(c_int), parameter :: STREAM_ALL = NUM_STREAMS\n");

    fprintf(DSLHEADER, "typedef int Stream;\n\n");

    // Reduction types
    fprintf(DSLHEADER, "#define AC_FOR_RTYPES(FUNC)\\\n");
    fprintf(DSLHEADER, "FUNC(%s)\\\n", "RTYPE_MAX");
    fprintf(DSLHEADER, "FUNC(%s)\\\n", "RTYPE_MIN");
    fprintf(DSLHEADER, "FUNC(%s)\\\n", "RTYPE_RMS");
    fprintf(DSLHEADER, "FUNC(%s)\\\n", "RTYPE_RMS_EXP");
    fprintf(DSLHEADER, "FUNC(%s)\\\n", "RTYPE_ALFVEN_MAX");
    fprintf(DSLHEADER, "FUNC(%s)\\\n", "RTYPE_ALFVEN_MIN");
    fprintf(DSLHEADER, "FUNC(%s)\\\n", "RTYPE_ALFVEN_RMS");
    fprintf(DSLHEADER, "FUNC(%s)\n", "RTYPE_SUM");

    size_t counter = 0;
    fprintf(FHEADER, "integer(c_int), parameter :: RTYPE_MAX = %lu\n", counter);
    ++counter;
    fprintf(FHEADER, "integer(c_int), parameter :: RTYPE_MIN = %lu\n", counter);
    ++counter;
    fprintf(FHEADER, "integer(c_int), parameter :: RTYPE_RMS = %lu\n", counter);
    ++counter;
    fprintf(FHEADER, "integer(c_int), parameter :: RTYPE_RMS_EXP = %lu\n", counter);
    ++counter;
    fprintf(FHEADER, "integer(c_int), parameter :: RTYPE_ALFVEN_MAX = %lu\n", counter);
    ++counter;
    fprintf(FHEADER, "integer(c_int), parameter :: RTYPE_ALFVEN_MIN = %lu\n", counter);
    ++counter;
    fprintf(FHEADER, "integer(c_int), parameter :: RTYPE_ALFVEN_RMS = %lu\n", counter);
    ++counter;
    fprintf(FHEADER, "integer(c_int), parameter :: RTYPE_SUM = %lu\n", counter);
    ++counter;
    fprintf(FHEADER, "integer(c_int), parameter :: NUM_REDUCTION_TYPES = %lu\n", counter);

    // Boundcond types
    fprintf(DSLHEADER, "#define AC_FOR_BCTYPES(FUNC)\\\n");
    fprintf(DSLHEADER, "FUNC(%s)\\\n", "BOUNDCOND_PERIODIC");
    fprintf(DSLHEADER, "FUNC(%s)\\\n", "BOUNDCOND_SYMMETRIC");
    fprintf(DSLHEADER, "FUNC(%s)\n", "BOUNDCOND_ANTISYMMETRIC");

    counter = 0;
    fprintf(FHEADER, "integer(c_int), parameter :: BOUNDCOND_PERIODIC = %lu\n", counter);
    ++counter;
    fprintf(FHEADER, "integer(c_int), parameter :: BOUNDCOND_SYMMETRIC = %lu\n", counter);
    ++counter;
    fprintf(FHEADER, "integer(c_int), parameter :: BOUNDCOND_ANTISYMMETRIC = %lu\n", counter);
    ++counter;

    // Defines for initial conditions in host_memory. Moved here to work with astaroth.conf
    fprintf(DSLHEADER, "#define AC_FOR_INIT_TYPES(FUNC)\\\n");
    fprintf(DSLHEADER, "FUNC(%s)\\\n", "INIT_TYPE_RANDOM");
    fprintf(DSLHEADER, "FUNC(%s)\\\n", "INIT_TYPE_XWAVE");
    fprintf(DSLHEADER, "FUNC(%s)\\\n", "INIT_TYPE_GAUSSIAN_RADIAL_EXPL");
    fprintf(DSLHEADER, "FUNC(%s)\\\n", "INIT_TYPE_ABC_FLOW");
    fprintf(DSLHEADER, "FUNC(%s)\\\n", "INIT_TYPE_SIMPLE_CORE");
    fprintf(DSLHEADER, "FUNC(%s)\\\n", "INIT_TYPE_KICKBALL");
    fprintf(DSLHEADER, "FUNC(%s)\\\n", "INIT_TYPE_VEDGE");
    fprintf(DSLHEADER, "FUNC(%s)\\\n", "INIT_TYPE_VEDGEX");
    fprintf(DSLHEADER, "FUNC(%s)\\\n", "INIT_TYPE_RAYLEIGH_TAYLOR");
    fprintf(DSLHEADER, "FUNC(%s)\n", "INIT_TYPE_RAYLEIGH_BENARD");

    counter = 0;
    fprintf(FHEADER, "integer(c_int), parameter :: INIT_TYPE_RANDOM = %lu\n", counter);
    ++counter;
    fprintf(FHEADER, "integer(c_int), parameter :: INIT_TYPE_XWAVE = %lu\n", counter);
    ++counter;
    fprintf(FHEADER, "integer(c_int), parameter :: INIT_TYPE_GAUSSIAN_RADIAL_EXPL = %lu\n",
            counter);
    ++counter;
    fprintf(FHEADER, "integer(c_int), parameter :: INIT_TYPE_ABC_FLOW = %lu\n", counter);
    ++counter;
    fprintf(FHEADER, "integer(c_int), parameter :: INIT_TYPE_SIMPLE_CORE = %lu\n", counter);
    ++counter;
    fprintf(FHEADER, "integer(c_int), parameter :: INIT_TYPE_KICKBALL = %lu\n", counter);
    ++counter;
    fprintf(FHEADER, "integer(c_int), parameter :: INIT_TYPE_VEDGE = %lu\n", counter);
    ++counter;
    fprintf(FHEADER, "integer(c_int), parameter :: INIT_TYPE_VEDGEX = %lu\n", counter);
    ++counter;
    fprintf(FHEADER, "integer(c_int), parameter :: INIT_TYPE_RAYLEIGH_TAYLOR = %lu\n", counter);
    ++counter;
    fprintf(FHEADER, "integer(c_int), parameter :: INIT_TYPE_RAYLEIGH_BENARD = %lu\n", counter);
    ++counter;

    // Fortran structs
    const char* fortran_structs = R"(
type, bind(C) :: AcMeshInfo
  integer(c_int), dimension(AC_NUM_INT_PARAMS)      :: int_params
  integer(c_int), dimension(AC_NUM_INT3_PARAMS, 3)  :: int3_params
  real, dimension(AC_NUM_REAL_PARAMS)               :: real_params
  real, dimension(AC_NUM_REAL3_PARAMS, 3)           :: real3_params
end type AcMeshInfo
  )";
    fprintf(FHEADER, "%s\n", fortran_structs);
}

static void
generate_library_hooks(void)
{
    size_t num_kernels = 0;
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        if (symbol_table[i].type_qualifier == KERNEL) {
            ++num_kernels;
            const char* id = symbol_table[i].identifier;
            fprintf(CUDAHEADER, "GEN_KERNEL_FUNC_HOOK(%s)\n", id);

            fprintf(KHEADER, "GEN_KERNEL_FUNC_DECL(%s)\n", id);
            fprintf(DHEADER, "GEN_DEVICE_FUNC_HOOK(%s)\n", id);

            fprintf(DSLHEADER,
                    "AcResult acDevice_%s(const Device device, const Stream stream, const int3 "
                    "start, const int3 end);\n",
                    id);
        }
    }

    // Generate user-facing enums
    fprintf(DSLHEADER, "typedef enum AcKernel {");
    bool first_element = true;
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        if (symbol_table[i].type_qualifier == KERNEL) {
            const char* id = symbol_table[i].identifier;
            if (first_element) {
                fprintf(DSLHEADER, "\n\tKERNEL_%s", id);
                first_element = false;
            }
            else {
                fprintf(DSLHEADER, ",\n\tKERNEL_%s", id);
            }
        }
    }
    fprintf(DSLHEADER, "\n} AcKernel;");

    // Generate kernel lookup table to connect enums with kernel-calling functions
    fprintf(KHEADER, "const ComputeKernel kernel_lookup[%ld] = {", num_kernels);
    first_element = true;
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        if (symbol_table[i].type_qualifier == KERNEL) {
            const char* id = symbol_table[i].identifier;
            if (first_element) {
                fprintf(KHEADER, "\n\tAC_KERNEL_FUNC_NAME(%s)", id);
                first_element = false;
            }
            else {
                fprintf(KHEADER, ",\n\tAC_KERNEL_FUNC_NAME(%s)", id);
            }
        }
    }
    fprintf(KHEADER, " };");
}

int
main(void)
{
    root = astnode_create(NODE_UNKNOWN, NULL, NULL);

    const int retval = yyparse();
    if (retval) {
        fprintf(stderr, "Fatal error: DSL compilation failed\n");
        return EXIT_FAILURE;
    }

    DSLHEADER  = fopen(dslheader_filename, "w+");
    CUDAHEADER = fopen(cudaheader_filename, "w+");
    KHEADER    = fopen(kheader_filename, "w+");
    DHEADER    = fopen(dheader_filename, "w+");
    FHEADER    = fopen(fheader_filename, "w+");
    assert(DSLHEADER);
    assert(CUDAHEADER);
    assert(KHEADER);
    assert(DHEADER);
    assert(FHEADER);

    // Add built-in param symbols
    for (size_t i = 0; i < ARRAY_SIZE(builtin_int_params); ++i)
        add_symbol(SYMBOLTYPE_OTHER, UNIFORM, INT, builtin_int_params[i]);

    // Add built-in params
    /*
    add_symbol(SYMBOLTYPE_OTHER, UNIFORM, INT, "AC_mxy");
    add_symbol(SYMBOLTYPE_OTHER, UNIFORM, INT, "AC_nxy");
    add_symbol(SYMBOLTYPE_OTHER, UNIFORM, INT, "AC_nxyz");
    add_symbol(SYMBOLTYPE_OTHER, UNIFORM, INT, "AC_xy_plate_bufsize");
    add_symbol(SYMBOLTYPE_OTHER, UNIFORM, INT, "AC_xz_plate_bufsize");
    add_symbol(SYMBOLTYPE_OTHER, UNIFORM, INT, "AC_yz_plate_bufsize");
    add_symbol(SYMBOLTYPE_OTHER, UNIFORM, INT3, "AC_global_grid_n");
    add_symbol(SYMBOLTYPE_OTHER, UNIFORM, INT3, "AC_multigpu_offset");
    */
    for (size_t i = 0; i < ARRAY_SIZE(builtin_int3_params); ++i)
        add_symbol(SYMBOLTYPE_OTHER, UNIFORM, INT3, builtin_int3_params[i]);

    for (size_t i = 0; i < ARRAY_SIZE(builtin_real_params); ++i)
        add_symbol(SYMBOLTYPE_OTHER, UNIFORM, SCALAR, builtin_real_params[i]);

    /*
    for (size_t i = 0; i < ARRAY_SIZE(builtin_real3_params); ++i)
        add_symbol(SYMBOLTYPE_OTHER, UNIFORM, VECTOR, builtin_real3_params[i]);
    */

    // Generate
    traverse(root);
    generate_headers();
    generate_preprocessed_structures();
    generate_library_hooks();

    // print_symbol_table();

    // Cleanup
    fclose(DSLHEADER);
    fclose(CUDAHEADER);
    fclose(KHEADER);
    fclose(DHEADER);
    fclose(FHEADER);
    astnode_destroy(root);

    fprintf(stdout, "-- Generated %s\n", dslheader_filename);
    fprintf(stdout, "-- Generated %s\n", cudaheader_filename);
    fprintf(stdout, "-- Generated %s\n", kheader_filename);
    fprintf(stdout, "-- Generated %s\n", dheader_filename);
    fprintf(stdout, "-- Generated %s\n", fheader_filename);

    return EXIT_SUCCESS;
}
:q
