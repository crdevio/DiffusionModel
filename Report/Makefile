# Variables
TEX=pdflatex
BIB=bibtex
MAIN=report

# Cibles par défaut
all: $(MAIN).pdf

# Compilation du fichier .tex
$(MAIN).pdf:
	$(TEX) $(MAIN).tex
	$(BIB) $(MAIN)
	$(TEX) $(MAIN).tex
	$(TEX) $(MAIN).tex

# Nettoyage des fichiers temporaires générés lors de la compilation
clean:
	rm -f *.aux *.log *.bbl *.blg *.toc *.out *.lof *.lot *.fls *.fdb_latexmk $(MAIN).pdf

# Cible pour forcer la re-compilation
rebuild: clean all

# Règle pour afficher l'aide
help:
	@echo "Makefile pour la compilation de report.tex avec references.bib"
	@echo "Commandes disponibles :"
	@echo "  all      - Compile le document (par défaut)"
	@echo "  clean    - Supprime les fichiers temporaires générés"
	@echo "  rebuild  - Nettoie et recompille le document"
	@echo "  help     - Affiche ce message d'aide"
