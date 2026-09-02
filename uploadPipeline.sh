git status
read -p "Enter file names you want to commit: " files
if [ "$files" == ""]; then
    echo "Aborting script..."
    exit 0
fi

git add $files
read -p "Do you want to commit any changes? (y/n): " commit_changes
if [ "$commit_changes" == "n" ]; then
    echo "Aborting script..."
    exit 0
elif [ "$commit_changes" == "y" ]; then
    echo "Continuing..."
else
    echo "Invalid input. Aborting script..."
    exit 1
fi

read -p "Enter commit message: " commit
git commit -m "$commit"
git pull
read -p "Do you want to push to remote? (y/n): " push
if [ "$push" == "n" ]; then
    echo "Aborting script..."
    exit 0
fi
git push
