TEXFILES= proposal.tex

all: pdf 

pdf: 
	pdflatex proposal
	bibtex	proposal
	pdflatex proposal
	pdflatex proposal

clean:
	rm -rf *.aux *.dvi *.bbl *.blg *.log *.pdf *.ps
	rm -rf proposal *~
	rm -rf core #*#
