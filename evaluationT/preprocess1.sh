#!/bin/bash 
 
if [ -z "$2" ];then

echo "evaluateWER.sh <hypothesis-CTM-file> <dev | test>"
exit 0
fi

hypothesisSTM=$1
partition=$2


cat ${hypothesisSTM} | sort  -k1,1 | sed -e 's/__LEFTHAND__ //g' -e 's/ __LEFTHAND__//g' -e 's/ __EPENTHESIS__//g' -e 's/__EPENTHESIS__ //g' -e 's/ __EMOTION__//g' -e 's/__EMOTION__ //g'| sed -e 's,\b__[^_ ]*__\b,,g' -e 's,\bloc-\([^ ]*\)\b,\1,g' -e 's,\bcl-\([^ ]*\)\b,\1,g' -e 's,\b\([^ ]*\)-PLUSPLUS\b,\1,g' -e 's,\b\([A-Z][A-Z]*\)RAUM\b,\1,g' -e 's,WIE AUSSEHEN,WIE-AUSSEHEN,g'  -e 's,^\([A-Z]\) \([A-Z][+ ]\),\1+\2,g' -e 's,[ +]\([A-Z]\) \([A-Z]\) , \1+\2 ,g'| sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|  sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +]SCH\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +]NN\) \([A-Z][ +]\),\1+\2,g'| sed -e 's,\([ +][A-Z]\) \(NN[ +]\),\1+\2,g'| sed -e 's,\([ +][A-Z]\) \([A-Z]\)$,\1+\2,g' | perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;' > ${partition}
