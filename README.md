The Pencil Code
---------------

The Pencil Code moved to
[GitHub](https://github.com/pencil-code/pencil-code)
on 19 April 2015. It was previously hosted at
[Google Code](https://code.google.com/p/pencil-code/).

In order to checkout the code with
[Subversion](https://subversion.apache.org), use the command
```sh
svn checkout https://github.com/pencil-code/pencil-code/trunk pencil-code --username <github-username>
```
where `<github-username>` is your GitHub username.

To get started, run one of the samples:
```sh
unix>  cd pencil-code
unix>  source sourceme.csh  [or . sourceme.sh]
unix>  cd samples/conv-slab
unix>  mkdir data
```
To set up the symbolic links and compile the code:
```sh
unix>  pc_setupsrc
unix>  pc_build  [ -f /path/to/config/file.conf ]
```
To create the initial condition and run the code:
```sh
unix>  pc_start  [ -f /path/to/config/file.conf ]
unix>  pc_run    [ -f /path/to/config/file.conf ]
```

See `pencil-code/config/hosts/*/*.conf` for sample config files. For more
details, see the manual in the `doc/` directory (also available
[here](http://pencil-code.nordita.org/)).

-----------------------------------------------------------------------------

If you are using bash and you do not want to "source sourceme.sh" on each
session, you can insert the following into your .bashrc and/or .bash_profile:
```sh
export PENCIL_HOME=$HOME/pencil-code  [or wherever you have the code]
_sourceme_quiet=1; . $PENCIL_HOME/sourceme.sh; unset _sourceme_quiet
```
If you are using csh insert the following into your .cshrc:
```sh
setenv PENCIL_HOME $HOME/pencil-code  [or wherever you have the code]
source $PENCIL_HOME/sourceme.csh
```

## Documentation

* The [manual][manual] is the main source of information

## How to contribute to the Pencil Code

* For all changes to the code, make sure the auto-test still runs

* If you have write access: check in your changes and make sure you can
  fix possible problems emerging on [travis-ci.com][travis] as well as the
  [hourly][hourly] and [daily][daily] auto-tests.

* If you have only read access: fork this repository and use pull requests to contribute.

## Code of Conduct

* The Pencil Code community adheres to the [conduct][Contributor Covenant Code of Conduct].
  Please familiarize yourself with its details.

## License

* The Pencil Code is under the [license][GNU public license agreement].

[travis]: https://www.travis-ci.com/github/pencil-code/pencil-code
[hourly]: http://norlx51.nordita.org/~brandenb/pencil-code/tests/gfortran_hourly.html
[daily]: http://norlx51.nordita.org/~brandenb/pencil-code/tests/g95_debug.html
[conduct]: https://github.com/pencil-code/pencil-code/blob/master/license/CODE_OF_CONDUCT.md
[manual]: https://github.com/pencil-code/website/blob/master/doc/manual.pdf
[license]: https://github.com/pencil-code/pencil-code/blob/master/license/GNU_public_license.txt

