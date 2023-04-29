# Contributing

Contributions to Astaroth are very welcome!

This document details how to create good contributions. There are two primary concerns: 

1. The codebase should stay maintainable and commits should adhere to a consistent style.
2. New additions should not disrupt the work of others.

## Basic workflow

> "There is something that needs fixing"

1. Create your work. See [Programming](#markdown-header-programming) and [Committing](#markdown-header-committing) .
2. When done, check that autotests still pass by running `./ac_run -t`.
3. **[Recommended]:** Autoformat your code. See [Formatting](#markdown-header-formatting).
4. Create a pull request.

## Programming
* **Strive for code clarity over micro-optimizations.**
    * Readability and simplicity should always be preferred over anything else outside of performance-critical parts.
    * Give variables meaningful names and add comments to parts that are not immediately clear from context.
* **Avoid breaking existing functionality.** 
    * Do not modify existing interface functions in any way. Bugfixes are exceptions. If you need new functionality, create a new function. 
    * Do not rename or redefine global variables or constants.
    
## Committing
* Prefer multiple small commits over few large ones.
* Provide meaningful commit messages.
* If a feature consists of multiple commits, consider creating a new branch. See [Managing feature branches](#markdown-header-managing-feature-branches) and [About branches in general](#markdown-header-about-branches-in-general) for more details. When done, issue the pull request to the new branch.

## Formatting

If you have `clang-format`, you may run `scripts/fix_style.sh`. This script will recursively fix style of all the source files down from the current working directory. The script will ask for a confirmation before making any changes.

> **WARNING** The script will replace old source files with new formatted versions. Ensure that you have committed your changes before running `fix_style.sh` to be safe.

Basic rules:

- Use [K&R indentation style](https://en.wikipedia.org/wiki/Indentation_style#K&R_style) and 4 space tabs. 
- Line width is 100 characters.
- Start function names after a linebreak in source files. 
- [Be generous with `const` type qualifiers](https://isocpp.org/wiki/faq/const-correctness). 
- When in doubt, see [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

## Header example:
```cpp
// Licence notice and doxygen description here
#pragma once
#include "avoid_including_headers_here.h"

/** Doxygen comments */
void globalFunction(void);
```


## Source example:
```cpp
#include "parent_header.h"

#include <standard_library_headers.h>

#include "other_headers.h"
#include "more_headers.h"

typedef struct {
	int data;
} SomeStruct;

static inline int small_function(const SomeStruct& stuff) { return stuff.data; }

// Pass constant structs always by reference (&) and use const type qualifier.
// Modified structs are always passed as pointers (*), never as references.
// Constant parameters should be on the left-hand side, while non-consts go to the right.
static void
local_function(const SomeStruct& constant_struct, SomeStruct* modified_struct)
{
	modified_struct->data = constant_struct.data;
}

void
globalFunction(void)
{
	return;
}
```

## Managing feature branches 

1. Ensure that you're on the latest version of master. `git checkout master && git pull`

2. Create a feature branch with `git checkout -b <feature_name_year-month-date>`, f.ex. `git checkout -b forcingtests_2019-01-01`

3. Do your commits in that branch until your new feature works

4. Merge master with your feature branch `git merge master`

5. Resolve the conflicts and test that the code compiles and still works by running `./ac_run -t`

6. If everything is OK, commit your final changes to the feature branch and merge it to master `git commit && git checkout master && git merge <your feature branch> && git push`

7. Unless you really have to keep your feature branch around for historical/other reasons, remove it from remote by calling `git push origin --delete <your feature branch>`

A flowchart is available at [doc/commitflowchart.png](https://bitbucket.org/jpekkila/astaroth/src/2d91df19dcb3/doc/commitflowchart.png?at=master).

## About branches in general

* Unused branches should not kept around after merging them into master in order to avoid cluttering the repository. 

* `git branch -a --merged` shows a list of branches that have been merged to master and are likely not needed any more.

* `git push origin --delete <feature branch>` deletes a remote branch while `git branch -d <feature branch>` deletes a local branch

* If you think that you have messed up and lost work, run `git reflog` which lists the latests commits. All work that has been committed should be accessible with the hashes listed by this command with `git checkout <reflog hash>`.













