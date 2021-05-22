@ECHO OFF

rd "source\api" /s /q
rd "build\" /s /q

make html
