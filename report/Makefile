TEXFILES= report.tex

all: pdf 

pdf: 
	pdflatex report
	bibtex	report
	pdflatex report
	pdflatex report

clean:
	rm -rf *.aux *.dvi *.bbl *.blg *.log *.pdf *.ps
	rm -rf report *~
	rm -rf core #*#
