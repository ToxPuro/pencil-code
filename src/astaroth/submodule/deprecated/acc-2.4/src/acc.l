%option yylineno
%option noyywrap
%option noinput
%option nounput

D [0-9]
L [a-zA-Z_]

%{
#include "acc.tab.h"
%}

%%

"Scalar"    { return SCALAR; } /* Builtin types */
"Vector"    { return VECTOR; }
"Matrix"    { return MATRIX; }
"void"      { return VOID; } /* Rest of the types inherited from C */
"int"       { return INT; }
"int3"      { return INT3; }
"Complex"   { return COMPLEX; }
"ScalarField" { return SCALARFIELD; }
"VectorField" { return VECTOR; }
"ScalarArray" { return SCALARARRAY; }

"Kernel"       { return KERNEL; } /* Function specifiers */
"Device"       { return DEVICE; }
"Preprocessed" { return PREPROCESSED; }

"const"     { return CONSTANT; }
"in"        { return IN; } /* Device func storage specifiers */
"out"       { return OUT; }
"uniform"   { return UNIFORM; }

"else if"   { return ELIF; }
"if"        { return IF; }
"else"      { return ELSE; }
"for"       { return FOR; }
"while"     { return WHILE; }

"return"    { return RETURN; }

{D}+"."{D}+         { return REAL_NUMBER; } /* Literals */
{D}+"."{D}+d+       { return DOUBLE_NUMBER; }
{D}+"."{D}+f+       { return NUMBER; }
{D}+[lu]*           { return NUMBER; }
{L}({L}|{D})*       { return IDENTIFIER; }
\"(.)*\"            { return IDENTIFIER; } /* String */

"&&"                { return LAND; } /* Logical operations */
"||"                { return LOR; }

[<>/*+-=]"="        { return BINARY_OP; } /* Generic binary operations */

"++"                { return INPLACE_INC; }
"--"                { return INPLACE_DEC; }

[-+*/;=\[\]{}(),\.<>|&] { return yytext[0]; } /* Characters */


"//".*              { /* Skip regular comments */ }
[ \t\n\v\r]+        { /* Ignore whitespace, tabs and newlines */ }
.                   { printf("unrecognized char %d: [%c]\n", *yytext, *yytext); }


%%
