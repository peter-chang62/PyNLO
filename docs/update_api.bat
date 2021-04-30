@ECHO OFF

rd "source\api" /s /q

setlocal

set SPHINX_APIDOC_OPTIONS=show-inheritance

sphinx-apidoc --no-toc --module-first --templatedir="source\_templates" -o "source\api" "..\pynlo"

endlocal