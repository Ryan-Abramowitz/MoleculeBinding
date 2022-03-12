# MoleculeBinding
ML to predict which molecules bind to a protein (Rice SWEET2, https://www.rcsb.org/structure/5CTH)

in case of git errors: https://stackoverflow.com/questions/11706215/how-can-i-fix-the-git-error-object-file-is-empty


find .git/objects/ -type f -empty | xargs rm
git fetch -p
git fsck --full
