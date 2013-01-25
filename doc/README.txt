These docs are built using sphinx with  `make html` or `make latexpdf`. 
Once built, the docs exist in `build/html/` and `build/latex/`.

The docs can be uploaded to github autmatically, using the gh-pages.py 
script. `python gh-pages [tag]`, where `tag` is a release tag or `dev`.
An example session

    git checkout v0.13
    make html 
    make latexpdf
    python gh-pages.py
    cd gh-pages
    git push
