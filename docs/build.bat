@ECHO OFF

rd "source\api" /s /q
rd "build\html" /s /q

make html
