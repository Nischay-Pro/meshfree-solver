if success(`cmp -s ../backup_results/primvals_cuda48738_64_1000.txt ../results/primvals_cuda48738_64_1000.txt`)
    println("Integrity verified")
else
    println("Integrity failed")
end