#!/bin/bash
# Install missing LaTeX packages for Elsevier CAS template (cas-dc)
# Run with: sudo bash do.sh

PACKAGES=(
  makecell
  multirow
  sttools
  footmisc
  xstring
  moreverb
  preprint
  wrapfig
  stix
  charissil
  inconsolata
)

echo "=== Installing packages for Elsevier CAS template ==="
echo "Packages to install: ${PACKAGES[*]}"
echo ""

for pkg in "${PACKAGES[@]}"; do
  echo "--- Installing $pkg ---"
  tlmgr install "$pkg" 2>&1
  if [ $? -eq 0 ]; then
    echo "  ✓ $pkg installed successfully"
  else
    echo "  ✗ Failed to install $pkg (may already be installed or unavailable)"
  fi
done

echo ""
echo "=== Done. ==="
echo ""
echo "--- Compile PDF (cas-dc double column) ---"
echo "pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex"
echo ""
echo "--- Generate DOCX (pandoc) ---"
echo "pandoc main-docx.tex --from latex --to docx --citeproc --bibliography=references.bib --number-sections --standalone -o \"Learning Rate Engineering.docx\""
