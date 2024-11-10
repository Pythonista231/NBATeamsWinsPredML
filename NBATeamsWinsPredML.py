import time
import sys
import numpy as np
from statistics import mean 
from scipy.stats import pearsonr
from scipy.stats import spearmanr
# from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


import tkinter as tk 
from tkinter.ttk import *
from ttkbootstrap import Style
from tkinter import Canvas

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC 
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains



# lables on training data are from 2023-2024 nba season. therefore features which are from the previous season are from 2022-2023 season. 
#prediction of the win percentage a team will have using machine learning. 





## three functions which will be used for testing the models.

def calculateSumSquaredResiduals(predictedYValues, trainingSetYValues): #function to calculate and return the sum of the square residuals. 
    residuals = trainingSetYValues - predictedYValues
    squaredResiduals = residuals ** 2
    return np.sum(squaredResiduals)


def testLinearRegressionModel(weights, testDataFormatted, yValues): #function to test any linear regression model with testing data that is in the correct format (matrix vstacked, with constants column). returns MSE & MAE. 
    #calc predicted values:
    predictedValues = testDataFormatted @ weights

    #calculate mse and mae: 
    mse = calculateSumSquaredResiduals(predictedValues, yValues)
    sumAbsoluteResiduals = 0
    for index, predictedValue in enumerate(predictedValues): 
        sumAbsoluteResiduals += abs(predictedValue - yValues[index])
    mae = sumAbsoluteResiduals / len(yValues)

    return mse, mae


def testPolyModel(model, testDataFormatted, yValues): # function to test any polynomial model with testing data that is in the correct format (matrix with constants column). returns the mse and mae. 
    # calculate the predicted values: 
    predictedValues = model.predict(testDataFormatted)


    #calculate mse and mae: 
    mse = calculateSumSquaredResiduals(predictedValues, yValues)
    sumAbsoluteResiduals = 0
    for index, predictedValue in enumerate(predictedValues): 
        sumAbsoluteResiduals += abs(predictedValue - yValues[index])
    mae = sumAbsoluteResiduals / len(yValues)
    
    return mse, mae







## data engineering (training data): 

nbaTeamsOrder = np.array([ 
                  "Atlanta Hawks", 
                  "Boston Celtics",
                  "Brooklyn Nets", 
                  "Charlotte Hornets", 
                  "Chicago Bulls", 
                  "Cleveland Cavaliers", 
                  "Detroit Pistons", 
                  "Indiana Pacers", 
                  "Miami Heat", 
                  "Milwaukee Bucks", 
                  "New York Knicks", 
                  "Orlando Magic", 
                  "Philadelphia 76ers", 
                  "Toronto Raptors", 
                  "Washington Wizards", 

                  "Dallas Mavericks", 
                  "Denver Nuggets", 
                  "Golden State Warriors", 
                  "Houston Rockets", 
                  "Los Angeles Clippers", 
                  "Los Angeles Lakers", 
                  "Memphis Grizzlies", 
                  "Minnesota Timberwolves", 
                  "New Orleans Pelicans", 
                  "Oklahoma City Thunder", 
                  "Phoenix Suns", 
                  "Portland Trail Blazers", 
                  "Sacramento Kings", 
                  "San Antonio Spurs", 
                  "Utah Jazz",])
teamNames = np.array([teamName.split()[-1] for teamName in nbaTeamsOrder])

## features: 
lastSeasonNumWinsTrainX = np.array([41, 57, 45, 27, 40, 51, 17, 35, 44, 58, 47, 34, 54, 41, 35, 38, 53, 44, 22, 44, 43, 51, 42, 42, 40, 45, 33, 48, 22, 37]) #the teams number of wins last season (2022-23 season).

lastSeasonSeedTrainX = np.array([7, 2, 6, 14, 10, 4, 15, 11, 8, 1, 5, 13, 3, 9, 12, 11, 1, 6, 14, 5, 7, 2, 8, 9, 10, 4, 13, 3, 15, 12]) # the team's seed last season in their conference (eastern conference or western conference). 2022-23 season (season before). 

numAllStarsTrainX = np.array([0, 3, 0, 0, 1, 1, 0, 2, 1, 2, 1, 0, 1, 0, 0, 2, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 0, 2, 0, 1,]) # this season's number of all stars per team (2023-24)



def showErrorAndDetails(driver, e): #this is a helper function that will be used by the two web scraping functions below to handle what happens when an error with web scraping happens. 
    print("\nAn error occured while trying to do web scraping, this is unusual.")
    print(f"Error details: {e}")
    print("Quit and rerun the program\n")
    driver.quit()
    sys.exit()

# to gather the average game rating of every team's top 5 players (usually starting 5). using selenium to web scrape. 
def getAllAverageRatingOfTop5Players(seasonChosen): # function to get the average rating of the top 5 players of an nba team at the season chosen - web scraping. seasonChosen needs to be in this format: 2023-24. returns a list of all the avg ratings of top 5 players in order of the teams on nbaTeamsOrder list. 

    def waitUntilSpinnerInvisible(): #a helper function that will be used by this function to wait for spinner (loading animation) to disappear. 
        WebDriverWait(driver, 20).until(EC.invisibility_of_element_located((By.CSS_SELECTOR, ".spinner-wrapper")))

    averageTop5RatingList = [] #as list for now, will be populated by web scraping, will convert to ndarray after function is called. 
    service = Service('C:/Users/micha/Downloads/msedgedriver.exe')
    options = webdriver.EdgeOptions()
    driver = webdriver.Edge(service=service, options = options)
    driver.delete_all_cookies()
    try: 
        driver.get(f"https://www.lineups.com/nba/roster/atlanta-hawks")
    except Exception as e: 
        showErrorAndDetails(driver, e)
    time.sleep(3)


    # clicking sort by rating once the button is clickable. 
    ratingBtn = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "th.text-center mfsorter[by='lineups_rating']")))
    ratingBtn.click()
    waitUntilSpinnerInvisible()
    time.sleep(0.2) # a little wait to make sure the players are sorted by rating (desc). 


    for teamName in nbaTeamsOrder: # for each team: 

        print(teamName, "is the one we got to")


        #scroll all the way up
        driver.execute_script("window.scrollTo(0, 0);")
    
        # we dont have to sort it by rating again since stays sorted for all teams. 



        # selecting the correct NBA team: 

        # click the teams dropdown menu once it is clickable. 
        time.sleep(0.2)
        teamsDropdownMenu = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.ID, "page-select-dropdown-teams")))
        teamsDropdownMenu.click() #click
        waitUntilSpinnerInvisible()

        # click a team once the dropdown menu is fully expanded
        WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".grouped-dropdown.show.dropdown")))        
        teamOptionToClick = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, f"//a[contains(@class, 'dropdown-item')]//img[@alt='{teamName}']")))
        teamOptionToClick.click() 
        waitUntilSpinnerInvisible()

        # making sure the team was selected before proceeding. 
        img = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, f"#page-select-dropdown-teams img[alt='{teamName}']")))
        time.sleep(0.2)



        

        # selecting the correct NBA season: 

        # click the seasons dropdown menu once it is clickable. 
        seasonsDropdownMenu = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.ID, "main-dropdown")))
        
        seasonsDropdownMenu.click() 
        waitUntilSpinnerInvisible()


        # click on a season once dropdown menu fully expanded. 
        WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CLASS_NAME, "dropdown-menu.show")))            
        
        seasonOptionToClick = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, f"//button[contains(@class, 'dropdown-item')]//span[text()='{seasonChosen}']")))
        seasonOptionToClick.click()  
        waitUntilSpinnerInvisible()

        
        #make sure the season was selected before proceeding. 
        seasonSelectedShown = False
        numberOfTries = 0
        while not seasonSelectedShown and numberOfTries < 10:
            span = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.grouped-dropdown.dropdown #main-dropdown span")))
            print(span.text) 
            if span.text == seasonChosen: 
                seasonSelectedShown = True
            else: 
                numberOfTries += 1
                time.sleep(0.3)
        if not seasonSelectedShown: 
            print("\nError with web scraping, unusal, the right NBA season wasn't able to be chosen, quit the program and rerun\n")
            driver.quit()
            sys.exit()




        time.sleep(0.2)
        # getting the ratings of the top 5 players (usually the starting lineup). 
        top5RatingsList = []
        for i in range(5): 
            playerRows = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table.multi-row-data-table.t-stripped tbody tr")))
            playerRow = playerRows[i]
            ratingDiv = playerRow.find_element(By.CSS_SELECTOR, "app-rating div")
            rating = int(ratingDiv.text)
            top5RatingsList.append(rating)
        averageTop5RatingList.append(mean(top5RatingsList))

    driver.quit()
    return averageTop5RatingList


print("Starting web scraping to gather average rating of top 5 players of every team (training data)")
average2k24RatingTop5List = getAllAverageRatingOfTop5Players('2023-24') #2k24 is the name of the basketball game. 2k24 represents 2023-24 season, which is the 'current season' as far as the training data is concerned. 
# average2k24RatingTop5List = [84.6, 86, 80.6, 78.8, 84.4, 84.6, 77.6, 81.4, 85.4, 86.6, 81.6, 78.4, 85.4, 79.8, 79.6, 83, 83.4, 87.8, 81, 85.6, 86.2, 85, 85, 84.4, 80.4, 86.6, 83.4, 80.4, 75.8, 78.4]
average2k24RatingTop5TrainX = np.array(average2k24RatingTop5List)
print("\n\n\n")



# to get the team's current in-game rating, will use web scraping (selenium). 
def getAllTeamRatings(seasonChosenFullFormat): # function to get the in game rating of every nba team. seasonChosenFullFormat needs to be in this format: 2023-2024. returns a list with all the team ratings in order of nbaTeamOrder list. will be turned into ndarray after function called. 

    service = Service('C:/Users/micha/Downloads/msedgedriver.exe')
    driver = webdriver.Edge(service=service)
    
    url = f'https://hoopshype.com/nba2k/teams/{seasonChosenFullFormat}/'
    teamRatingDict = {}
    
    try: 
        driver.get(url)
    except Exception as e: 
        showErrorAndDetails(driver, e)


    # waiting until the tbody containing the rows is present: 
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'tbody'))
    )


    # then proceed to find the trows. 
    teamRows = driver.find_elements(By.CSS_SELECTOR, 'table.hh-salaries-ranking-table.hh-salaries-table-sortable tbody tr')
    

    # Iterate over each row and extract data
    for teamRow in teamRows:
        try: 
            teamRating = teamRow.find_element(By.CSS_SELECTOR, 'td.value.hh-salaries-sorted').get_attribute('data-value')
            teamName = teamRow.find_element(By.CSS_SELECTOR, 'td.name a').text
            teamRatingDict[teamName] = teamRating
        except Exception as e: 
            showErrorAndDetails(driver, e)
    
    driver.quit()

    # now to change the dictionary into a list of elements with the same order as the nbaTeamsOrder list. 
    teamRatingsList = []
    for team in nbaTeamsOrder: 
        # adding the team's rating to the list: 
        teamRatingsList.append(float(teamRatingDict[team]))


    return teamRatingsList

print("\n\nStarting web scraping to gather rating of every team (testing data)")
teamRatings2k24List = getAllTeamRatings('2023-2024')
# teamRatings2k24List = [76.4, 79.6, 76.5, 75.1, 77.3, 77.1, 75.6, 77.2, 75.8, 77.6, 76.8, 75.3, 77.2, 76.4, 74.2, 75.8, 75.4, 77.7, 77.0, 77.4, 78.9, 75.5, 76.4, 76.7, 76.0, 77.9, 74.6, 76.4, 74.9, 75.3]
teamRatings2k24TrainX = np.array(teamRatings2k24List)
print("\n\n\n")



#target variable would like to predict: 

percentageWinsTrainY = np.array([
    0.439, 0.780, 0.390, 0.256, 0.476, 0.564, 0.171, 0.567, 0.539, 0.580,
    0.600, 0.562, 0.562, 0.305, 0.183, 0.621, 0.681, 0.554, 0.500, 0.602,
    0.562, 0.329, 0.681, 0.568, 0.685, 0.570, 0.256, 0.560, 0.268, 0.378
])









## data engineering (test data): 
# since the training data treats 2023-24 as the current nba season with all the training data corresponding to that season, the test data will consider 2022-23 season as the current season (e.g. data that is about this season will be from 22-23 season and data that is about 'last season' will be from 21-22 season)

lastSeasonNumWinsTestX = np.array([43, 51, 44, 43, 46, 44, 23, 25, 53, 51, 37, 22, 51, 48, 35, 52, 48, 53, 20, 42, 33, 56, 46, 36, 24, 64, 27, 30, 34, 49]) # 2021-22 season. since 2022-23 is considered 'this season' for test data. 

lastSeasonSeedTestX = np.array([9, 2, 7, 10, 6, 8, 14, 13, 1, 3, 11, 15, 4, 5, 12, 4, 6, 3, 15, 8, 11, 2, 7, 9, 14, 1, 13, 12, 10, 5]) # 2021-22 season since 2022-23 is considered 'this season' for test data. 

print("\n\nStarting web scraping to gather average rating of top 5 players of every team (testing data)")
average2k23RatingTop5List = getAllAverageRatingOfTop5Players('2022-23')
# average2k23RatingTop5List = [85.2, 85.6, 81.4, 81.4, 84.6, 84.6, 78.6, 79.6, 85.8, 86.4, 82, 78.4, 85.8, 82.8, 82.2, 85.4, 83.4, 86.4, 78.6, 83.6, 85.2, 84.8, 84.6, 84.4, 79.6, 88.4, 82.2, 80.4, 76.8, 77]
average2k23RatingTop5TestX = np.array(average2k23RatingTop5List)

print("\n\nStarting web scraping to gather rating of every team (testing data)")
teamRatings2k23List = getAllTeamRatings('2022-2023')
# teamRatings2k23List = [76.8, 78.6, 76.5, 75.6, 76.4, 77.2, 74.2, 75.9, 76.2, 76.1, 75.8, 74.6, 76.2, 74.9, 76.1, 75.3, 75.1, 76.3, 73.7, 76.4, 75.0, 75.2, 75.7, 77.4, 74.4, 77.0, 75.1, 75.9, 73.2, 74.3]
teamRatings2k23TestX = np.array(teamRatings2k23List)

teamRatings2k23TestX = np.array([77.7, 76.9, 78.2, 75.5, 75.8, 75.5, 75.6, 77.2, 76.6, 77.1, 77.6, 75.4, 77.0, 75.7, 76.2, 76.7, 76.4, 76.8, 75.1, 76.7, 79.4, 75.1, 76.5, 75.5, 74.2, 78.1, 76.7, 75.5, 74.9, 76.3]) # this is the 2k22 rating of the team itself. (2021-22 season's game is 2k22)

numAllStarsTestX = np.array([1, 1, 2, 1, 2, 0, 0, 0, 1, 2, 0, 0, 2, 1, 0, 1, 1, 3, 0, 0, 1, 1, 1, 0, 0, 2, 0, 0, 1, 2]) # 27 all stars in the 2021-22. there were 26 all stars in total in the 22-23 season, but shouldn't make a big difference. 


percentageWinsTestY = np.array([0.524, 0.622, 0.537, 0.524, 0.561, 0.537, 0.280, 0.305, 0.646, 0.622, 0.451, 0.268, 0.622, 0.585, 0.427, 0.634, 0.585, 0.646, 0.244, 0.512, 0.402, 0.683, 0.561, 0.439, 0.293, 0.780, 0.329, 0.366, 0.415, 0.598])












## checking correlation between the feature variables and the target variable: (correlation coefficient & spearman's correlation coefficent)


def showCoefficients(var1, descriptionOfVar1, var2 = percentageWinsTrainY): # function to calculate and show the correlation coefficient & spearman's corr coefficient of this feature and the target var, as well as p val of both. 

    print(descriptionOfVar1 + " (feature)"+ " & % wins (target variable)")


    #(pearson) corr coef: 
    correlationCoefficient, pVal = pearsonr(var1, var2)
    print(f"->Corr coeff: {correlationCoefficient}")
    print(f"    P val: {pVal}")
    
    #spearman's correlation coefficient: 
    spearmansCorr, pVal = spearmanr(var1, var2)
    print(f"->Spearman's correlation coeff: {spearmansCorr}")
    print(f"    P val: {pVal}")

    print("\n")



print("\n\nCorrelation coefficient tests the linear relationship (between the feature and target variable) ")
print("Spearman's correlation coefficient tests how well the relationship between the feature and target variable can be described using a monotonic function:\n")

showCoefficients(lastSeasonNumWinsTrainX, "Number of wins last season")

showCoefficients(lastSeasonSeedTrainX, "Last season seed")

showCoefficients(average2k24RatingTop5TrainX, "Average in-game rating of top 5 players (usually starting 5) this season")

showCoefficients(teamRatings2k24TrainX, "In-game rating of the team this season")

showCoefficients(numAllStarsTrainX, "Number of all stars in the team this season")











## data visualisation. 

def draw3DGraph(master): #a function used by showDataVisualisation function. 
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    
    # Color map based on percentageWinsTrainY
    colors = cm.viridis(percentageWinsTrainY / max(percentageWinsTrainY))
    
    ax1.scatter(lastSeasonNumWinsTrainX, average2k24RatingTop5TrainX, percentageWinsTrainY, 
                c=colors, marker='o')

    ax1.set_xlabel('Number of wins last season')
    ax1.set_ylabel('Average in-game rating of top 5 players (usually starting 5) this season')
    ax1.set_zlabel('% Wins')
    ax1.set_title("3D visualisation of 2 key features and target variable.")

    # Add labels to each point of the team's name
    for i in range(len(nbaTeamsOrder)):
        ax1.text(lastSeasonNumWinsTrainX[i], average2k24RatingTop5TrainX[i], 
                 percentageWinsTrainY[i], teamNames[i], size=8, color='k')

    # Embed the 3D plot in the specified master frame
    canvas3D = FigureCanvasTkAgg(fig1, master=master)
    canvas3D.draw()
    canvas3D.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)



def showDataVisualisation(): # plotting each of the features with the output variable and one interactive 3d graph showing 2 key features (number of wins last season and average in-game rating of the team) & the output vairable. 
    root = tk.Tk()
    root.title("DATA VISUALISATION")

    #frame
    frame = Frame(root)
    frame.pack(fill=tk.BOTH,expand=True)


    #scrollbar: 
    scrollbarVertical = Scrollbar(frame, orient = tk.VERTICAL)
    scrollbarVertical.pack(side=tk.RIGHT, fill=tk.Y)

    canvas = Canvas(frame, yscrollcommand=scrollbarVertical.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(-1 * (event.delta // 120), "units"))

    plotFrame = Frame(canvas)

    
    titleLabel = Label(text = "Visualisation of each feature with target vairable, close window to continue", master = canvas, font = 'Calibri 26 underline bold', wraplength=700, anchor="center",  justify="center").pack(pady = 65)
    canvas.create_window((0, 0), window=plotFrame, anchor="nw")



    fig = plt.figure(figsize = (16, 23)) 
    gs = fig.add_gridspec(3, 2, height_ratios = [1, 1, 1.2])

    # fig.subplots_adjust(wspace=0.5)  # padding


    featuresArrays = [lastSeasonNumWinsTrainX,lastSeasonSeedTrainX, average2k24RatingTop5TrainX, teamRatings2k24TrainX, numAllStarsTrainX]
    xLabels = ["Number of wins last season", "Last season seed", "Average in-game rating of top 5 players (usually starting 5) this season", "In-game rating of the team this season", "Number of all stars in the team this season"]

    titles = [xLabel + " & % wins this season" for xLabel in xLabels]

    # first plot the 5 2d graphs: 
    axs = []
    for i in range(5):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        ax.scatter(featuresArrays[i], percentageWinsTrainY, marker='o', color='blue')
        ax.set_xlabel(xLabels[i])
        ax.set_ylabel("Percentage wins this season")
        ax.set_title(titles[i])
        axs.append(ax)

    # replace the 6th subplot with the 3d plot: 
    ax3D = fig.add_subplot(gs[2, 1], projection='3d')

    # Color map based on percentageWinsTrainY
    colors = cm.viridis(percentageWinsTrainY / max(percentageWinsTrainY))

    # 3D Scatter plot
    scatter = ax3D.scatter(
        lastSeasonNumWinsTrainX, 
        average2k24RatingTop5TrainX, 
        percentageWinsTrainY, 
        c=colors, 
        marker='o'
    )

    ax3D.set_xlabel('Number of wins last season')
    ax3D.set_ylabel('Average in-game rating of top 5 players')
    ax3D.set_zlabel('% Wins')
    ax3D.set_title("Interactive 3D Visualisation of Key Features and Target Variable")

    # Add labels to each point in the 3D scatter plot
    for i in range(len(nbaTeamsOrder)):
        ax3D.text(
            lastSeasonNumWinsTrainX[i], 
            average2k24RatingTop5TrainX[i], 
            percentageWinsTrainY[i], 
            teamNames[i], 
            size=11, 
            color='k'
        )


    canvasFig = FigureCanvasTkAgg(fig, master= plotFrame)
    canvasFig.draw()
    canvasFig.get_tk_widget().pack(side= tk.TOP, fill = tk.BOTH, expand = True)
    
    
    # update scrollregion:
    plotFrame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))
    
    root.geometry('1570x1700')
    root.lift()
    root.attributes("-topmost", True)
    root.mainloop()




showDataVisualisation() #this shows the user 6 graphs to visualise the data. 










## multiple linear regression: ( the most appropriate model based on the linearity of the data from the correlation coefficients and the graphs too. 
# not sure whether to include last season seed feature as well as avg top 5 players rating this season, since they had low correlatoins for both spearman and pearson, will investigate and decide whether or not to include. 

# made 3 different models - all 5 feautres, 4 features without last season seed, 4 features without top 5 players rating 2k24. 

dataMatrixLinReg = np.vstack([lastSeasonNumWinsTrainX,lastSeasonSeedTrainX, average2k24RatingTop5TrainX, teamRatings2k24TrainX, numAllStarsTrainX, np.ones(len(numAllStarsTrainX))]).T
dataMatrixLinRegWithoutSeed = np.vstack([lastSeasonNumWinsTrainX, average2k24RatingTop5TrainX, teamRatings2k24TrainX, numAllStarsTrainX, np.ones(len(numAllStarsTrainX))]).T 
dataMatrixLinRegWithoutAvgTop5Ratings = np.vstack([lastSeasonNumWinsTrainX,lastSeasonSeedTrainX, teamRatings2k24TrainX, numAllStarsTrainX, np.ones(len(numAllStarsTrainX))]).T

# Calculate weights (coefficients) using least squares for each of the 3. 
resultsLinReg = np.linalg.lstsq(dataMatrixLinReg, percentageWinsTrainY, rcond=None)
resultsLinRegWithoutSeed = np.linalg.lstsq(dataMatrixLinRegWithoutSeed, percentageWinsTrainY, rcond=None)
resultsLinRegWithoutAvgTop5Ratings = np.linalg.lstsq(dataMatrixLinRegWithoutAvgTop5Ratings, percentageWinsTrainY, rcond = None)

# weights for the multiple linear regression model with the last season seed (my default linear regression model)
weightsListLinReg = resultsLinReg[0]
weightsListLinRegWithoutSeed = resultsLinRegWithoutSeed[0]
weightsListLinRegWithoutAvgTop5Ratings=  resultsLinRegWithoutAvgTop5Ratings[0]
w1, w2, w3, w4, w5, c = weightsListLinReg

# outputting/describing the weights of the first model (all 5 features): 
print("As per the multiple linear regression model: ")
print(f"An increase of last season's number of wins by 1 will give the team {float(w1) * 100}% more wins this season ")
print(f"An increase of last season's seed by 1 position will give the team {float(w2) * 100}% more wins this season")
print(f"An increase of this season's in-game rating by 1 point of the team will give the team {float(w3) * 100}% more wins this season")
print(f"An increase of the average in-game rating of the team's top 5 players by 1 point will give the team {float(w4) * 100}% more wins this season")
print(f"One more all star will give the team {float(w4) * 100}% more wins this season\n\n")








## checking & showing the perfomrance of the multiple linear regression model with & witout 2 different features to decide whether or not to include them. 
# since both last season seed and average rating of top 5 players features has a RELATIVELY lower pearson corr coeff, testing whether or not to include them: 

# first checking & showing how the 2 models fit to training data (sum of square residuals to the training data). 
sumSquareResidualsLinReg = resultsLinReg[1]
print(f"MULTIPLE LINEAR REGRESSION: Model fit to training data (with all 5 features): The sum of the square of residuals compared to training data:  {sumSquareResidualsLinReg[0] * 100}% wins per season. ")
sumSquareResidualsLinRegWithoutSeed = resultsLinRegWithoutSeed[1]
print(f"MULTIPLE LINEAR REGRESSION: Model fit to training data (without last season seed feature): The sum of the square of residuals compared to training data: {sumSquareResidualsLinRegWithoutSeed[0] * 100}% wins per season")
sumSquareResidualsLinRegWithoutAvgTop5Ratings = resultsLinRegWithoutAvgTop5Ratings[1]
print(f"MULTIPLE LINEAR REGRESSION: Model fit to training data (without last average rating top 5 players feature): The sum of the square of residuals compared to training data: {sumSquareResidualsLinRegWithoutAvgTop5Ratings[0] * 100}% wins per season \n\n")

# and finally showing the performance of the 3 models by using the test data: 
# formatting test data: 
dataMatrixLinRegTest = np.vstack([lastSeasonNumWinsTestX,lastSeasonSeedTestX, average2k23RatingTop5TestX, teamRatings2k23TestX, numAllStarsTestX, np.ones(len(numAllStarsTestX))]).T
dataMatrixLinRegWithoutSeedTest = np.vstack([lastSeasonNumWinsTestX, average2k23RatingTop5TestX, teamRatings2k23TestX, numAllStarsTestX, np.ones(len(numAllStarsTestX))]).T 
dataMatrixLinRegWithoutAvgTop5RatingsTest = np.vstack([lastSeasonNumWinsTestX,lastSeasonSeedTestX, teamRatings2k23TestX, numAllStarsTestX, np.ones(len(numAllStarsTestX))]).T

mseLinRegTest, maeLinRegTest = testLinearRegressionModel(weightsListLinReg, dataMatrixLinRegTest, percentageWinsTestY)
mseLinRegWithoutSeedTest, maeLinRegWithoutSeedTest = testLinearRegressionModel(weightsListLinRegWithoutSeed, dataMatrixLinRegWithoutSeedTest, percentageWinsTestY)
mseLinRegWithoutAvgTop5RatingsTest, maeLinRegWithoutAvgTop5RatingsTest = testLinearRegressionModel(weightsListLinRegWithoutAvgTop5Ratings, dataMatrixLinRegWithoutAvgTop5RatingsTest, percentageWinsTestY)

# outputting: 
print(f"MULTIPLE LINEAR REGRESSION: Model performance on test data (all 5 features): MAE:  {maeLinRegTest * 100}% wins per season, MSE:  {mseLinRegTest * 100}% wins per season. ")
print(f"MULTIPLE LINEAR REGRESSION: Model performance on test data (without last season seed): MAE:   {maeLinRegWithoutSeedTest * 100}% wins per season, MSE:  {mseLinRegWithoutSeedTest * 100}% wins per season. ")
print(f"MULTIPLE LINEAR REGRESSION: Model performance on test data (without average rating top 5 players feature): MAE:   {maeLinRegWithoutAvgTop5RatingsTest * 100}% wins per season, MSE:  {mseLinRegWithoutAvgTop5RatingsTest* 100}% wins per season. \n\n")

#----------------OUTPUT----------------:
# MULTIPLE LINEAR REGRESSION: Model performance on test data (all 5 features): MAE:  5.99019413049251% wins per season, MSE:  17.9781246882228% wins per season.
# MULTIPLE LINEAR REGRESSION: Model performance on test data (without last season seed): MAE:   6.031676988539445% wins per season, MSE:  17.453890362026627% wins per season.
# MULTIPLE LINEAR REGRESSION: Model performance on test data (without average rating top 5 players feature): MAE:   6.04722201330752% wins per season, MSE:  18.40928824015445% wins per season.
# -> multiple linear regression performed best with all 5 features, so that is the preferable mult lin reg model. 
#--------------------------------------









## trying RFE (recursive feature implementation) with multiple linear regression: 

multLinReg = LinearRegression() #multiple linear regression. 
rfe = RFE(estimator = multLinReg, n_features_to_select=4)
rfe.fit(dataMatrixLinReg, percentageWinsTrainY)


features = ["Number of wins last season", "Last season seed", "Average in-game rating of top 5 players this season", "In-game team rating this season", "Number of all stars in the team this season"]
print("Feature ranking: ", list(zip(features, rfe.ranking_)))

selectedFeatures = [featureName for featureName, rank in zip(features, rfe.ranking_) if rank == 1]
print(f"Selected features: {selectedFeatures}\n\n\n")

# the output received from this only gives 'average in game rating of top 5 players this season' a rank of 2. however this isn't helpful as previous checks suggested better performance on testing data by keeping all 5 features instead of removing the average in game rating of top 5 players feature. 















## now that model development has finished, accepting user input to use for inference: 

# taking user input: 
print("\n\n\nIn order for this to work, you have to pick a team which you would like to predict the % wins of in a certain season. ")
print("Treat that season as 'this season' so if asked about 'last season' enter the values of the one before your 'this season' in focus. ")
numberOfWinsLastSeasonInput = int(input("How many wins did the team have last season? (not a percentage, enter the absolute number of wins) "))
lastSeasonSeedInput = int(input("What was the seed of the team last season? "))
averageInGameRatingTop5PlayersInput = float(input("What is the average rating of the top 5 players of the team this season? "))
inGameTeamRatingInput = float(input("What is the in game rating of the team this season? "))
numAllStarsInput = int(input("How many all stars does the team have this season? "))

inferenceValues = np.array([numberOfWinsLastSeasonInput, lastSeasonSeedInput, averageInGameRatingTop5PlayersInput, inGameTeamRatingInput, numAllStarsInput, 1])

inferenceOutput = inferenceValues @ weightsListLinReg
print(f"For a team with {numberOfWinsLastSeasonInput} wins and {lastSeasonSeedInput} seed last season, with an in-game rating of {inGameTeamRatingInput} for the team and {averageInGameRatingTop5PlayersInput} for their top 5 players, with {numAllStarsInput} all stars, my proprietary machine learning model predicts it will have {inferenceOutput * 100}% wins this season!")







