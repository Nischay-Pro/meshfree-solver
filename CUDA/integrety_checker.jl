if success(`cmp -s ./backup_results/primvals_cuda9600_64_1000.txt ./backup_results/primvals_cuda9600_64_1000.txt`)
    println("Integrity verified")
else
    println("Integrity failed")
end