1. Download the .py files in one folder
2. Run the train_and_save_models file first
3. after you get a message in the terminal that the models have been saved, you can open file explorer and see where the 5 .pkl files were saved
4. copy path to those 5.pkl files and open fake_news_detector file
5. at the joblib load, you need to replace the file names with the path and make sure that u have this: (r"C:/user/directory/file.pkl")
6. it is in this format, the r before " is important
7. then run the fake news detector file and there u go, this will predict news headlines and tell u if they are fake or real along with the confidence level
