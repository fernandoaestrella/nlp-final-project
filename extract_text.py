from bs4 import BeautifulSoup
filename="C:\Users\ferna\Desktop\GRAD SCHOOL\Semester 3\131A-1  Introduction to Natural Language Processing with Python\Homeworks\Final Project\BIU-RPI-Event-Extraction-Project\ACE_EVENT\corpus\orig\nw\timex2normAPW_ENG_20030424.0698.sgm"
f = open(filename, "r")
contents = f.read()
print(contents)
# soup = BeautifulSoup()