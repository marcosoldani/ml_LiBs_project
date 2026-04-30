PDF := main.pdf
TEX := main.tex
OUT_DIR := ..

# Auto-discover TeX Live install on macOS so that `make pdf` works even when
# TeX Live's bin/ directory is not on the user's PATH (a common issue on
# fresh MacTeX / TeX Live installs).
TEXLIVE_BIN := $(firstword $(wildcard \
    /usr/local/texlive/2026/bin/universal-darwin \
    /usr/local/texlive/2025/bin/universal-darwin \
    /usr/local/texlive/2024/bin/universal-darwin \
    /Library/TeX/texbin \
    /opt/homebrew/bin))

LATEXMK := $(if $(TEXLIVE_BIN),PATH="$(TEXLIVE_BIN):$$PATH" latexmk,latexmk)

.PHONY: pdf clean clean-all watch

# Build the PDF and publish it next to the README in the parent dir.
pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(TEX)
	cp $(PDF) $(OUT_DIR)/$(PDF)
	@echo "PDF published → $(OUT_DIR)/$(PDF)"

watch:
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode $(TEX)

# Default clean: remove only intermediate build files, keep ../main.pdf.
clean:
	$(LATEXMK) -c
	rm -f *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.lof *.lot *.log
	rm -f *.out *.run.xml *.toc *.synctex.gz
	rm -f *.acn *.acr *.alg *.glg *.glo *.gls *.ist
	rm -f .DS_Store

# Hard reset: also remove the local PDF (the published one in ../ is untouched).
clean-all: clean
	rm -f $(PDF)
