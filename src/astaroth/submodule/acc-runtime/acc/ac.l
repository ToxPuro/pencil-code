%option yylineno
%option noyywrap
%option noinput
%option nounput

D [0-9]
L [a-zA-Z_]

%{
#include "tab.h"
%}

%%

"int"                   { return INT; } /* Type specifiers */
"uint"                  { return UINT; }
"int3"                  { return INT3; }
"real"                  { return REAL; }
"real3"                 { return REAL3; }
"Matrix"                { return MATRIX; }
"Field"                 { return FIELD; }
"Stencil"               { return STENCIL; }

"Kernel"                { return KERNEL; } /* Type qualifiers */
"Sum"                   { return SUM; }
"Max"                   { return MAX; }

"if"        { return IF; } /* Keywords */
"else if"   { return ELIF; }
"else"      { return ELSE; }
"while"     { return WHILE; }
"for"       { return FOR; }
"in"        { return IN; }
"break"     { return BREAK; }
"continue"  { return CONTINUE; }
"return"    { return RETURN; }

"hostdefine".*        { return HOSTDEFINE; }

{L}({L}|{D})*             { return IDENTIFIER; }   /* Literals */
\"(\\.|[^"\\])*\"         { return STRING; }       /* String literal */
[+-/*|&%^]?=              { return ASSIGNOP; }     /* Assignment */
(<<=)|(>>=)               { return ASSIGNOP; }     /* Assignment */
[/*-+<>|&%^]              { return BINARY_OP; }    /* Operators */
(&&)|(\|\|)|(<<)|(>>)     { return BINARY_OP; }    /* Operators */
(==)|(>=)|(<=)|(!=)|(^=)  { return BINARY_OP; }    /* Operators */
[-+*/:;=\[\]{}(),\.<>|&!] { return yytext[0]; }    /* Characters */
{D}+[.]{D}*               { return REALNUMBER; }
{D}+[.]?{D}*(e[-]?{D}+)   { return REALNUMBER; }
{D}+[.]?{D}*(e[-]?{D}+)?[d]  { return DOUBLENUMBER; } /* Numbers */
{D}+[.]?{D}*(e[-]?{D}+)?[f]?  { return NUMBER; }      /* Numbers */
0[xX][0-9a-fA-F]+[ulUL]*      { return NUMBER; }      /* Hex */



"#".*               { /* Skip regular comments */ }
[ \t\n\v\r]+        { /* Ignore whitespace, tabs and newlines */ }
.                   { fprintf(stderr, "unrecognized char %d: [%c]\n", *yytext, *yytext); }

%%

/*
[.]{D}+                   { return REALNUMBER; }
[.]?{D}+(e[-]?{D}+)       { return REALNUMBER; }
{D}?[.]+{D}*(e[-]?{D}+)?[f]?  { return NUMBER; }
{D}?[.]+{D}*(e[-]?{D}+)?[d]  { return DOUBLENUMBER; }
*/
