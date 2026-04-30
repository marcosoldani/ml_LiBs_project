# Tell latexmk to run `makeglossaries` after the first pdflatex pass so that
# acronyms (\gls, \acrshort, …) resolve correctly.
add_cus_dep('glo', 'gls', 0, 'makeglo2gls');
add_cus_dep('acn', 'acr', 0, 'makeglo2gls');

sub makeglo2gls {
    system("makeglossaries \"$_[0]\"");
}

# File extensions to clean with `latexmk -c`.
$clean_ext .= ' acn acr alg glg glo gls ist';
