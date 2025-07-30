1. Download the .py files in one folder
2. Run the train_and_save_models file first
3. After you get a message in the terminal that the models have been saved, you can open file explorer and see where the 5 .pkl files were saved
4. Copy path to those 5.pkl files and open fake_news_detector file
5. At the joblib load, you need to replace the file names with the path and make sure that you have this: (r"C:/user/directory/file.pkl")
6. It is in this format, the r before " is important
7. Then run the fake news detector file and there you go, this will predict news headlines and tell you if they are fake or real along with the confidence level
