#!/usr/bin/env python3
import os
import re
import argparse

def remove_duplicates(path, dry_run=False):
    seen = set()
    # Captura tudo até o último ".pdf", ignorando o sufixo randômico antes de ".md"
    pattern = re.compile(r'^(.*\.pdf)(.+)\.md$', re.IGNORECASE)

    for fname in os.listdir(path):
        fullpath = os.path.join(path, fname)
        if not os.path.isfile(fullpath) or not fname.lower().endswith('.md'):
            continue

        m = pattern.match(fname)
        if not m:
            continue

        base = m.group(1)
        if base in seen:
            if dry_run:
                print(f"[DRY-RUN] Encontrado duplicado: {fname}")
            else:
                print(f"Removendo duplicado: {fname}")
                os.remove(fullpath)
        else:
            seen.add(base)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove arquivos .md duplicados na pasta 'docs', considerando o prefixo até '.pdf'."
    )
    parser.add_argument(
        "directory", nargs="?", default="docs",
        help="Diretório com arquivos .md (padrão: docs)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Somente lista o que seria removido, sem apagar nada"
    )
    args = parser.parse_args()
    remove_duplicates(args.directory, args.dry_run)
