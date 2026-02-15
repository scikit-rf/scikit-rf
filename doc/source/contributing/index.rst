.. _contributing:
    :github_url:


.. image:: ../_static/scikit-rf-title-flat.svg
    :target: ../_static/scikit-rf-title-flat.svg
    :height: 100
    :align: center


Contributing to scikit-rf
==========================

Sponsoring the Project
----------------------
It is possible to sponsor the maintainers and developers of the scikit-rf package, using the GitHub Sponsor option ("Sponsor this project") visible on the `scikit-rf GitHub page <https://github.com/scikit-rf/scikit-rf>`_.

Sponsoring is one more way to contribute to open source: financially supporting the people who build and maintain it. Funding individuals helps them keep doing important work, expands opportunities to participate, and gives developers the recognition they deserve.


Contributing to the Code
------------------------

.. note:: if you feel that the instructions on this page are too complicated, but you still would like to contribute, do not hesitate to email us on the `scikit-rf mailing list <https://groups.google.com/forum/#!forum/scikit-rf>`_ or join us in the `scikit-rf Slack channel <https://join.slack.com/t/scikit-rf/shared_invite/zt-d82b62wg-0bdSJjZVhHBKf6687V80Jg>`_ or the `scikit-rf Matrix/Element channel <https://app.element.io/#/room/#scikit-rf:matrix.org>`_ to get some help.


**skrf** uses the "Fork + Pull" collaborative development model. If this is new to you, see GitHub's articles on  `forking <https://help.github.com/articles/fork-a-repo>`_ and `pull requests <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests>`_.

Basically, you will:

1. `Fork <https://help.github.com/articles/fork-a-repo>`_ the `GitHub scikit-rf repository <https://github.com/scikit-rf/scikit-rf>`_,

2. Make your modifications.

3. Then send a `pull request (PR) <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests>`_ for your additions to be merged into the main scikit-rf repository. Your proposal will be reviewed or discussed and you may receive some comments which only aim to make your contribution as great as possible!


.. tip:: When making your modification locally, you may need to work into a dedicated development environment in order to not interfere with the scikit-rf package that you have `already installed <../tutorials/Installation.html>`_. You can use for example `anaconda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_. In order for the anaconda environment to find your local scikit-rf repo, use the convenient `conda-develop <https://docs.conda.io/projects/conda-build/en/latest/resources/commands/conda-develop.html>`_ command.

Prek
++++

You can use prek_ to automate your code quality checks and perform automated fixes in **scikit-rf**.

To enable prek on your computer (make sure it is installed first), open a terminal and Navigate to
the :file:`scikit-rf/` directory that contains your clone of scikit-rf's repository, then run:

.. code-block:: bash

    prek install

.. note::

   Once prek has been installed for a repository, prek will
   run every time you try to commit a change. If any prek checks
   fail, or if prek changes any files, it will be necessary to
   redo `git add` on the changed files and `git commit` once
   again.

.. tip::

   To commit a change without running prek, use the `-n` or
   `--no-verify` flag with git.


Basic git command-line workflow
+++++++++++++++++++++++++++++++

The following is a basic example of the git commands that can be used to contribute to the code.

.. code-block:: sh

    # create your own fork of scikit-rf in the GitHub interface, say ME/scikit-rf

    # clone your new fork locally, using either:
    git clone ME@github.com:ME/scikit-rf.git

    # if you have ssh keys setup, or if not:
    git clone https://github.com/scikit-rf/scikit-rf.git

    # ... make your changes...

    # commit changes locally
    git commit -a

    # push changes to your repo
    git push origin

    # create a pull request on github.com


Staying Synchronized
++++++++++++++++++++

To keep your local version synchronized (up-to-date) with the scikit-rf repository, `add a "remote" (call it "upstream") <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/configuring-a-remote-for-a-fork>`_. From this remote, you can `"fetch" and "merge" the official scikit-rf repo's changes into your local repo <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork>`_.

.. code-block:: sh

    git remote add upstream https://github.com/scikit-rf/scikit-rf.git

    # Fetch any new changes from the original repo
    git fetch upstream

    # Merges any changes fetched into your working files
    git merge upstream/master



Tests
+++++

Tests are vital for software reliability and maintainability. Writing tests often require additional effort but saves time in the long run. Tests enable us to quickly discover when we introduce new errors. It is also a way to provide examples of how functions and classes were originally intended to be used.

Before making a Pull Request, we advise contributors to run the tests locally to check if nothing has been broken following their modifications. In addition, we highly recommend providing new tests when adding new features.

The structure of the testing generally follows the conventions of `numpy/scipy <https://github.com/numpy/numpy/blob/master/doc/TESTS.rst>`_. Test cases live in the module, or submodule, which they are testing, and are located in a directory called `tests`. So, the tests of the media module are located at `skrf/media/tests/`.
Tests can be run most easily with `pytest <https://docs.pytest.org/en/latest/index.html>`_.

You probably **don't** want to run the tests for the virtual instruments ``skrf.vi`` with the rest of the tests, so these tests are excluded by default.

To run all the tests (except the virtual instruments)

.. code-block:: sh

    cd scikit-rf
    # Create environment and install dependencies, needed only once.
    python -m venv .venv
    pip install -e .[test,visa,netw,xlsx,plot,docs,testspice] --compile

    # Activate Environment, needed for all following steps.
    # on Linux Systems
    . .venv/bin/activate

    # on Windows
    .\.venv\Scripts\activate

    pytest

To run all tests *and* all tutorials and example notebooks in your current environment (recommended before making a pull request):

.. code-block:: sh

    pytest --nbval-lax


If you want to test a single file or directory, you need to override the default pytest configuration and indicate the test path. For example, to run only the tests associated with the Network object (-v to increase the verbosity):

.. code-block:: sh

    pytest -v -c "" skrf/tests/test_network.py


It is also possible to select some particular tests using the regex option (-k):

.. code-block:: sh

    pytest -v -c "" skrf/calibration/tests/test_calibration.py -k "test_error_ntwk"





Contributing to the Documentation
----------------------------------

Examples and Tutorials
++++++++++++++++++++++

Usage examples of scikit-rf are welcomed, especially when adding new features. We are using `Jupyter Notebooks <https://jupyter.org/>`_ to write the examples and the tutorials, which are located in the ``scikit-rf/docs/source/examples/`` and ``doc/source/examples`` directories. These notebooks are then converted into webpages with the sphinx extension called `nbsphinx <http://nbsphinx.readthedocs.io/>`_.

The docs are automatically built and `served by readthedocs <https://scikit-rf.readthedocs.io/en/latest/>`_ when a Pull Request is accepted. The Python package requirements to build the docs are kept in ``scikit-rf/pyproject.toml``.

.. important:: Before pushing to your repo and making a pull request, at a minimum you will need to clear the notebook outputs using the "Clear All Output" command in the notebook (or install `nbstripout <https://pypi.python.org/pypi/nbstripout>`_ so that the output is not tracked in git (or the repo size would grow infinitely).


Reference (API) or static documentation
+++++++++++++++++++++++++++++++++++++++

The documentation source files can be found in ``doc/source/``.

The reference documentation for the functions, classes, and submodules are documented in docstrings following the conventions put forth by `Numpy/Scipy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`_. The documentation as a whole is generated using sphinx, and  written using reStructed (.rst) Text.

.. tip:: If you want to write some .rst file yourself, please use an RST format editor and checker (ex: `<https://livesphinx.herokuapp.com/>`_), as Sphinx is (very) picky with the syntax...


Building the documentation locally
++++++++++++++++++++++++++++++++++

Before making a pull request concerning the documentation, it is a good idea to test locally if your changes lead to the desired HTML output (sometimes some problems can arise during the conversion to HTML). The documentation is built by the following commands:

.. code-block:: sh

    # be sure to be in the scikit-rf/doc directory
    make html


The built docs then reside in ``doc/build/html``.




Join the **scikit-rf** team!
----------------------------

Do you like using scikit-rf? `Merchandise is available for you to express your love <https://scikit-rf.org/merch.html>`_.

.. image:: https://raw.githubusercontent.com/scikit-rf/scikit-rf/master/logo/skrfshirtwhite.png
    :height: 400
    :align: center
