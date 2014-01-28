

notebooks:
	for file in tutorials/*.ipynb; do\
	  runipy  --pylab -o $$file;\
	done
	cd tutorials;ipython nbconvert --to rst *.ipynb
	sed -i 's/``/`/g' tutorials/*.rst


