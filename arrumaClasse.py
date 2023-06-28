import glob
import sys

arquivos=glob.glob(f"{sys.argv[1]}/*.txt")

for i in arquivos:
    with open(i, "r") as arquivo:
        conteudo = arquivo.read()
        parts=conteudo.split(" ")
    parts[0]=sys.argv[2]
    newconteudo=' '.join(parts)
    print(newconteudo)
    with open(i, "w") as arquivo:
        arquivo.write(newconteudo)