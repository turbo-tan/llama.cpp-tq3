#!/usr/bin/env bash

if [ $# -lt 2 ]; then
    printf "Usage: $0 <git-repo> <target-folder> [<test-exe>]\n"
    exit 1
fi

if [ $# -eq 3 ]; then
    toktest=$3
else
    toktest="./test-tokenizer-0"
fi

if [ ! -x $toktest ]; then
    printf "Test executable \"$toktest\" not found!\n"
    exit 1
fi

repo=$1
folder=$2

if [ -d $folder ] && [ -d $folder/.git ]; then
    (cd $folder; git pull)
else
    git clone $repo $folder
fi

if git lfs version >/dev/null 2>&1; then
    (cd $folder; git lfs pull)
fi

# byteswap models if on big endian
if [ "$(uname -m)" = s390x ]; then
    for f in $folder/*/*.gguf; do
        echo YES | python3 "$(dirname $0)/../gguf-py/gguf/scripts/gguf_convert_endian.py" $f big
    done
fi

find "$folder" -type f -name '*.gguf' -print0 | while IFS= read -r -d '' gguf; do
    if [ -f "$gguf.inp" ] && [ -f "$gguf.out" ]; then
        "$toktest" "$gguf"
    else
        printf "Found \"$gguf\" without matching inp/out files, ignoring...\n"
    fi
done
