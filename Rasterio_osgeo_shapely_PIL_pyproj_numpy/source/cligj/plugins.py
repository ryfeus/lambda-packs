"""
Common components required to enable setuptools plugins.

In general the components defined here are slightly modified or subclassed
versions of core click components.  This is required in order to insert code
that loads entry points when necessary while still maintaining a simple API
is only slightly different from the click API.  Here's how it works:

When defining a main commandline group:

    >>> import click
    >>> @click.group()
    ... def cli():
    ...    '''A commandline interface.'''
    ...    pass

The `click.group()` decorator turns `cli()` into an instance of `click.Group()`.
Subsequent commands hang off of this group:

    >>> @cli.command()
    ... @click.argument('val')
    ... def printer(val):
    ...    '''Print a value.'''
    ...    click.echo(val)

At this point the entry points, which are just instances of `click.Command()`,
can be added to the main group with:

    >>> from pkg_resources import iter_entry_points
    >>> for ep in iter_entry_points('module.commands'):
    ...    cli.add_command(ep.load())

This works but its not very Pythonic, is vulnerable to typing errors, must be
manually updated if a better method is discovered, and most importantly, if an
entry point throws an exception on completely crashes the group the command is
attached to.

A better time to load the entry points is when the group they will be attached
to is instantiated.  This requires slight modifications to the `click.group()`
decorator and `click.Group()` to let them load entry points as needed.  If the
modified `group()` decorator is used on the same group like this:

    >>> from pkg_resources import iter_entry_points
    >>> import cligj.plugins
    >>> @cligj.plugins.group(plugins=iter_entry_points('module.commands'))
    ... def cli():
    ...    '''A commandline interface.'''
    ...    pass

Now the entry points are loaded before the normal `click.group()` decorator
is called, except it returns a modified `Group()` so if we hang another group
off of `cli()`:

    >>> @cli.group(plugins=iter_entry_points('other_module.commands'))
    ... def subgroup():
    ...    '''A subgroup with more plugins'''
    ...    pass

We can register additional plugins in a sub-group.

Catching broken plugins is done in the modified `group()` which attaches instances
of `BrokenCommand()` to the group instead of instances of `click.Command()`.  The
broken commands have special help messages and override `click.Command.invoke()`
so the user gets a useful error message with a traceback if they attempt to run
the command or use `--help`.
"""


import os
import sys
import traceback
import warnings

import click


warnings.warn(
    "cligj.plugins has been deprecated in favor of click-plugins: "
    "https://github.com/click-contrib/click-plugins. The plugins "
    "module will be removed in cligj 1.0.",
    FutureWarning, stacklevel=2)


class BrokenCommand(click.Command):

    """
    Rather than completely crash the CLI when a broken plugin is loaded, this
    class provides a modified help message informing the user that the plugin is
    broken and they should contact the owner.  If the user executes the plugin
    or specifies `--help` a traceback is reported showing the exception the
    plugin loader encountered.
    """

    def __init__(self, name):

        """
        Define the special help messages after instantiating `click.Command()`.

        Parameters
        ----------
        name : str
            Name of command.
        """

        click.Command.__init__(self, name)

        util_name = os.path.basename(sys.argv and sys.argv[0] or __file__)

        if os.environ.get('CLIGJ_HONESTLY'):  # pragma no cover
            icon = u'\U0001F4A9'
        else:
            icon = u'\u2020'

        self.help = (
            "\nWarning: entry point could not be loaded. Contact "
            "its author for help.\n\n\b\n"
            + traceback.format_exc())
        self.short_help = (
            icon + " Warning: could not load plugin. See `%s %s --help`."
            % (util_name, self.name))

    def invoke(self, ctx):

        """
        Print the error message instead of doing nothing.

        Parameters
        ----------
        ctx : click.Context
            Required for click.
        """

        click.echo(self.help, color=ctx.color)
        ctx.exit(1)  # Defaults to 0 but we want an error code


class Group(click.Group):

    """
    A subclass of `click.Group()` that returns the modified `group()` decorator
    when `Group.group()` is called.  Used by the modified `group()` decorator.
    So many groups...

    See the main docstring in this file for a full explanation.
    """

    def __init__(self, **kwargs):
        click.Group.__init__(self, **kwargs)

    def group(self, *args, **kwargs):

        """
        Return the modified `group()` rather than `click.group()`.  This
        gives the user an opportunity to assign entire groups of plugins
        to their own subcommand group.

        See the main docstring in this file for a full explanation.
        """

        def decorator(f):
            cmd = group(*args, **kwargs)(f)
            self.add_command(cmd)
            return cmd

        return decorator


def group(plugins=None, **kwargs):

    """
    A special group decorator that behaves exactly like `click.group()` but
    allows for additional plugins to be loaded.

    Example:

        >>> import cligj.plugins
        >>> from pkg_resources import iter_entry_points
        >>> plugins = iter_entry_points('module.entry_points')
        >>> @cligj.plugins.group(plugins=plugins)
        ... def cli():
        ...    '''A CLI aplication'''
        ...    pass

    Plugins that raise an exception on load are caught and converted to an
    instance of `BrokenCommand()`, which has better error handling and prevents
    broken plugins from taking crashing the CLI.

    See the main docstring in this file for a full explanation.

    Parameters
    ----------
    plugins : iter
        An iterable that produces one entry point per iteration.
    kwargs : **kwargs
        Additional arguments for `click.Group()`.
    """

    def decorator(f):

        kwargs.setdefault('cls', Group)
        grp = click.group(**kwargs)(f)

        if plugins is not None:
            for entry_point in plugins:
                try:
                    grp.add_command(entry_point.load())

                except Exception:
                    # Catch this so a busted plugin doesn't take down the CLI.
                    # Handled by registering a dummy command that does nothing
                    # other than explain the error.
                    grp.add_command(BrokenCommand(entry_point.name))
        return grp

    return decorator
