with open("../res/logs/hardy-weinberg-samuel.txt", "r", encoding="utf-8") as f1, open("../res/logs/hardy-weinberg-adam.txt", "r", encoding="utf-8") as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()

if lines1 == lines2:
    print("✅ Files are identical")
else:
    print("❌ Files differ")
