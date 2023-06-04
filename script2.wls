#!/usr/bin/env wolframscript
(* ::Package:: *)

Run["sed -i 's/\[ //g' result2.dat"]

Run["sed -i 's/\[//g' result2.dat"]

Run["sed -i 's/\]/\\n/g' result2.dat"]

Run["sed -i 's/      /\\n/g' result2.dat"]

Run["sed -i 's/   /\\n/g' result2.dat"]

Run["sed -i 's/  /\\n/g' result2.dat"]

Run["sed -i 's/ /\\n/g' result2.dat"]
