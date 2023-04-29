/*
    Nodes for the Abstract Syntax Tree

    Statement: syntactic unit tha expresses some action.
    May have internal components, expressions, which are evaluated

    Statements: return value
                block
*/
#include <assert.h>
#include <stdlib.h>

#define BUFFER_SIZE (4096)

#define GEN_ID(X) X
#define GEN_STR(X) #X

// clang-format off
#define FOR_NODE_TYPES(FUNC) \
    FUNC(NODE_UNKNOWN), \
    FUNC(NODE_DEFINITION), \
    FUNC(NODE_GLOBAL_DEFINITION), \
    FUNC(NODE_ITERATION_STATEMENT), \
    FUNC(NODE_DECLARATION), \
    FUNC(NODE_DECLARATION_LIST), \
    FUNC(NODE_ARRAY_DECLARATION), \
    FUNC(NODE_TYPE_DECLARATION), \
    FUNC(NODE_TYPE_QUALIFIER), \
    FUNC(NODE_TYPE_SPECIFIER), \
    FUNC(NODE_IDENTIFIER), \
    FUNC(NODE_FUNCTION_DEFINITION), \
    FUNC(NODE_FUNCTION_DECLARATION), \
    FUNC(NODE_COMPOUND_STATEMENT), \
    FUNC(NODE_FUNCTION_PARAMETER_DECLARATION), \
    FUNC(NODE_MULTIDIM_SUBSCRIPT_EXPRESSION), \
    FUNC(NODE_REAL_NUMBER), \
    FUNC(NODE_FOR_EXPRESSION)
// clang-format on

typedef enum { FOR_NODE_TYPES(GEN_ID), NUM_NODE_TYPES } NodeType;

typedef struct astnode_s {
    int id;
    struct astnode_s* parent;
    struct astnode_s* lhs;
    struct astnode_s* rhs;
    NodeType type; // Type of the AST node
    char* buffer;  // Indentifiers and other strings (empty by default)

    int token;   // Type of a terminal (that is not a simple char)
    int prefix;  // Tokens. Also makes the grammar since we don't have
    int infix;   // to divide it into max two-child rules
    int postfix; // (which makes it much harder to read)
} ASTNode;

static inline ASTNode*
astnode_create(const NodeType type, ASTNode* lhs, ASTNode* rhs)
{
    ASTNode* node = calloc(1, sizeof(node[0]));

    static int id_counter = 0;
    node->id              = id_counter++;
    node->type            = type;
    node->lhs             = lhs;
    node->rhs             = rhs;
    node->buffer          = NULL;

    node->token  = 0;
    node->prefix = node->infix = node->postfix = 0;

    if (lhs)
        node->lhs->parent = node;

    if (rhs)
        node->rhs->parent = node;

    return node;
}

static inline void
astnode_set_buffer(const char* buffer, ASTNode* node)
{
    node->buffer = strdup(buffer);
}

static inline void
astnode_destroy(ASTNode* node)
{
    if (node->lhs)
        astnode_destroy(node->lhs);
    if (node->rhs)
        astnode_destroy(node->rhs);
    if (node->buffer)
        free(node->buffer);
    free(node);
}

static inline void
astnode_print(const ASTNode* node)
{
    const char* node_type_names[] = {FOR_NODE_TYPES(GEN_STR)};

    printf("%s (%p)\n", node_type_names[node->type], node);
    printf("\tid:      %d\n", node->id);
    printf("\tparent:  %p\n", node->parent);
    printf("\tlhs:     %p\n", node->lhs);
    printf("\trhs:     %p\n", node->rhs);
    printf("\tbuffer:  %s\n", node->buffer);
    printf("\ttoken:   %d\n", node->token);
    printf("\tprefix:  %d ('%c')\n", node->prefix, node->prefix);
    printf("\tinfix:   %d ('%c')\n", node->infix, node->infix);
    printf("\tpostfix: %d ('%c')\n", node->postfix, node->postfix);
}

extern ASTNode* root;
