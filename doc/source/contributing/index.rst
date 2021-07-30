.. _contributing:
    :github_url:


.. image:: ../_static/scikit-rf-title-flat.svg
    :target: scikitrf_
    :height: 100
    :align: center


Contributing to scikit-rf
==========================

Sponsoring the Project
----------------------
It is possible to sponsor the maintainers and developpers of the scikit-rf package, using the GitHub Sponsor option ("Sponsor this project") visible in the `scikit-rf GitHub page <https://github.com/scikit-rf/scikit-rf>`_. 

Sponsoring is one more way to contribute to open source: financially supporting the people who build and maintain it. Funding individuals helps them keep doing important work, expands opportunities to participate, and gives developers the recognition they deserve.


Contributing to the Code
------------------------

.. note:: if you feel that the instructions on this page are too complicated, but you still would like to contribute, do not hesitate email us on the `scikit-rf mailing list <https://groups.google.com/forum/#!forum/scikit-rf>`_ or join us in the `scikit-rf Slack channel <https://join.slack.com/t/scikit-rf/shared_invite/zt-d82b62wg-0bdSJjZVhHBKf6687V80Jg>`_ to get some help.


**skrf** uses the "Fork + Pull" collaborative development model. If this new to you, see GitHub's articles on  `forking <https://help.github.com/articles/fork-a-repo>`_ and `pull requests <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests>`_. 

Basically, you will:

1. `Fork <https://help.github.com/articles/fork-a-repo>`_ the `GitHub scikit-rf repository <https://github.com/scikit-rf/scikit-rf>`_, 

2. Make your modifications. 

3. Then send a `pull request (PR) <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests>`_ for your additions to be merged into the main scikit-rf repository. Your proposal will be reviewed or discussed and you may receive some comments which only aim to make your contribution as great as possible!


.. tip:: When making your modification locally, you may need to work into a dedicated development environment in order to not interfere with the scikit-rf package that you have `already installed <https://scikit-rf.readthedocs.io/en/latest/tutorials/Installation.html>`_. You can use for example `anaconda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_. In order for the anaconda environment to find your local scikit-rf repo, use the convenient `conda-develop <https://docs.conda.io/projects/conda-build/en/latest/resources/commands/conda-develop.html>`_ command.


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

To keep your local version synchronized (up-to date) with the scikit-rf repository, `add a "remote" (call it "upstream") <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/configuring-a-remote-for-a-fork>`_. From this remote, you can `"fetch" and "merge" the official scikit-rf repo's changes into your local repo <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork>`_.

.. code-block:: sh

    git remote add upstream https://github.com/scikit-rf/scikit-rf.git

    # Fetch any new changes from the original repo
    git fetch upstream
    
    # Merges any changes fetched into your working files
    git merge upstream/master



Tests
+++++

Tests are vital for software reliability and maintainability. Writing tests often requires additional efforts, but saves time in the long run. Tests enable us to quickly discover when we introduce new errors. It is also a way to provide examples of how functions and classes were originally intended to be used.

Before making a Pull Request, we advise contributors to run the tests locally to check if nothing has been broken following their modifications. In addition, we highly recommend to provide new tests when adding new features.

The structure of the testing generally follows the conventions of `numpy/scipy <https://github.com/numpy/numpy/blob/master/doc/TESTS.rst.txt>`_. Test cases live in the module, or submodule, which they are testing, and are located in a directory called `tests`. So, the tests of the media module are located at `skrf/media/tests/`. 
Tests can be run most easily with `nosetest <http://nose.readthedocs.org/en/latest/>`_. 

You probably **don't** want to run the tests for the virtual instruments ``skrf.vi`` with the rest of the tests, so to prevent this, also install `nose-exclude <https://pypi.python.org/pypi/nose-exclude>`_ via pip (``pip install nose-exclude``) or conda. 

To run all the tests (except the virtual instruments)

.. code-block:: sh

    cd scikit-rf
    nosetests skrf -c nose.cfg

Or, to run test a single module or single test, 

.. code-block:: sh

    nosetests media/
    # ...
    nosetests tests/test_network.py
    # ...
    

Contributing to the Documentation
----------------------------------

Examples and Tutorials
++++++++++++++++++++++

Usage examples of scikit-rf are welcomed, especially when adding new features. We are using `Jupyter Notebooks <https://jupyter.org/>`_ to write the examples and the tutorials, which are located in the ``scikit-rf/docs/source/examples/`` and ``doc/source/examples`` directories. These notebooks are then converted into webpages with the sphinx extension called `nbsphinx <http://nbsphinx.readthedocs.io/>`_. 

The docs are automatically built and `served by readthedocs <https://scikit-rf.readthedocs.io/en/latest/>`_ when a Pull Request is accepted. The python package requirements to build the docs are kept in ``scikit-rf/requirements.txt``. 

.. important:: Before pushing to your repo and making a pull request, at a minimum you will need to clear the notebook outputs using the "Clear All Output" command in the notebook (or install `nbstripout <https://pypi.python.org/pypi/nbstripout>`_ so that the output is not tracked in git (or the repo size would grow infinitely). 


Reference (API) or static documentation
+++++++++++++++++++++++++++++++++++++++

The documentation source files can be found in ``doc/source/``. 

The reference documentation for the functions, classes, and submodules are documented in docstrings following the conventions put forth by `Numpy/Scipy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`_. The documentation as a whole is generated using sphinx, and  written using reStructed (.rst) Text. 

.. tip:: If you want to write some .rst file yourself, please use a RST format editor and checker (ex: `<https://livesphinx.herokuapp.com/>`_), as Sphinx is (very) picky with the syntax...


Building the documentation locally
++++++++++++++++++++++++++++++++++

Before making a pull request concerning the documentation, it is a good idea to test locally if your changes lead to the desired html output (sometimes some problems can arise during the conversion to html). The documentation is built by the following commands:

.. code-block:: sh

    # be sure to be in the scikit-rf/doc directory
    make html


The built docs then reside in ``doc/build/html``.




Join the **scikit-rf** team!
----------------------------

.. image:: https://raw.githubusercontent.com/scikit-rf/scikit-rf/master/logo/skrfshirtwhite.png
    :height: 400
    :align: center
