#!/usr/bin/env bash
dir=${1:-docs}
cd "$dir" || exit 1

for base in $(ls *.md | sed -E 's/(.*\.pdf).*/\1/' | sort -u); do
  # array de arquivos que começam com o prefixo base e terminam em .md
  files=( "$base"*".md" )
  # remove todos, exceto o primeiro
  for ((i=1; i<${#files[@]}; i++)); do
    echo "Removendo ${files[i]}"
    rm -- "${files[i]}"
  done
done
