import numpy as np
from statistics import mean 
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.preprocessing import PolynomialFeatures

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


# 24.2% with multiple linear regression in the previous one. 




# lables on training data are from 2023-2024 nba season. therefore features which are from the previous season are from 2022-2023 season. 
# predicting the number of wins a team will have based on the number of all starts they have as well as the average game rating of their starting 5. 
# using multiple linear regression at first and then transitioning to polynomial regression. 



# data engineering: 

nbaTeamsOrder = np.array([ "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls", 
                  "Cleveland Cavaliers", "Detroit Pistons", "Indiana Pacers", "Miami Heat", "Milwaukee Bucks", 
                  "New York Knicks", "Orlando Magic", "Philadelphia 76ers", "Toronto Raptors", "Washington Wizards", 
                  "Dallas Mavericks", "Denver Nuggets", "Golden State Warriors", "Houston Rockets", "Los Angeles Clippers", 
                  "Los Angeles Lakers", "Memphis Grizzlies", "Minnesota Timberwolves", "New Orleans Pelicans", 
                  "Oklahoma City Thunder", "Phoenix Suns", "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Utah Jazz",])
teamNames = np.array([teamName.split()[-1] for teamName in nbaTeamsOrder])

## features: 
lastYearNumWinsTrainX = np.array([41, 57, 45, 27, 40, 51, 17, 35, 44, 58, 47, 34, 54, 41, 35, 38, 53, 44, 22, 44, 43, 51, 42, 42, 40, 45, 33, 48, 22, 37]) #the teams number of wins last year (2022-23 season).

lastYearSeedTrainX = np.array([7, 2, 6, 14, 10, 4, 15, 11, 8, 1, 5, 13, 3, 9, 12, 11, 1, 6, 14, 5, 7, 2, 8, 9, 10, 4, 13, 3, 15, 12]) # the team's seed last year in their conference (eastern conference or western conference). 2022-23 season (season before). 

numAllStarsTrainX = np.array([0, 3, 0, 0, 1, 1, 0, 2, 1, 2, 1, 0, 1, 0, 0, 2, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 0, 2, 0, 1,]) # this season's number of all stars per team (2023-24)

inGameTeamRatingTrainX = np.array([76.4, 79.6, 76.5, 75.1, 77.3, 77.1, 75.6, 77.2, 75.8, 77.6,76.8, 75.3, 77.2, 76.4, 74.2, 75.8, 75.4, 77.7, 77.0, 77.4, 78.9, 75.5, 76.4, 76.7, 76.0, 77.9, 74.6, 76.4, 74.9, 75.3 ]) # the in-game (2k24) rating of the team (2023-2024 season - current season. )


# extracting the average game rating of the team's starting 5. for this, I am going to use the NBA 2k25 game since they have player ratings. 
# this isn't from the 2023-2024 season we are looking at, but rather one after it (24-25), however this feature is very highly correlated with the % wins in 23-24 so i still decided to include it. There shouldn't be a huge difference between the average starting 5 rating in 23-24 season and hte 24-25 season. Not perfect but better than not including it. 
def getAvgTop5PlayersRating(): 
    averageGameRatingSt5TrainX = []
    teamNamesForUrls = ['-'.join(team.strip().split()) for team in nbaTeamsOrder]
    service = Service('C:/Users/micha/Downloads/msedgedriver.exe')
    driver = webdriver.Edge(service=service)

    for teamNameForUrl in teamNamesForUrls: 
        url = f'https://www.2kratings.com/teams/{teamNameForUrl}'
        driver.get(url)
        
        ratingDivs = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.rating-updated"))
        )

        # Extract the 'data-order' attribute from the span inside each of the first 5 divs
        top5Ratings = []
        for div in ratingDivs[:5]:
            span = div.find_element(By.TAG_NAME, "span")  # Find the span within each div
            top5Ratings.append(int(float(span.get_attribute("data-order"))))  # Get the data-order attribute
        averageGameRatingSt5TrainX.append(mean(top5Ratings))

    driver.quit()
    return averageGameRatingSt5TrainX #returns a list. 

# print(getAvgTop5PlayersRating())
averageGameRatingSt5TrainX = np.array([81.6, 89.4, 79.2, 81.2, 80.6, 84.2, 80.6, 84.6, 83.8, 86.2, 86.2, 83, 86.2, 81.2, 80.4, 86.4, 85.6, 83.2, 82.6, 82.8, 86, 83.6, 85, 85, 86.4, 86, 81.2, 84.6, 82, 80.6]) #hard coded the values because next season/year the url i used to web scrape will show next year's data so will be less fitting with rest of training data. 


#target variable we would like to predict: 

percentageWinsTrainY = np.array([
    0.439, 0.780, 0.390, 0.256, 0.476, 0.564, 0.171, 0.567, 0.539, 0.580,
    0.600, 0.562, 0.562, 0.305, 0.183, 0.621, 0.681, 0.554, 0.500, 0.602,
    0.562, 0.329, 0.681, 0.568, 0.685, 0.570, 0.256, 0.560, 0.268, 0.378
])

print(len(percentageWinsTrainY))




## checking correlation between the features variables and the target variable: 

# firstly correlation coefficient. 
print("Testing the linear relationship between two of the features and the % wins")
correlationCoefficient, pVal = pearsonr(lastYearNumWinsTrainX, percentageWinsTrainY)
print("Number of wins last season:")
print("Corr coeff:", correlationCoefficient)
print("P val:", pVal)

print("Testing the linear relationship between two of the features and the % wins")
correlationCoefficient, pVal = pearsonr(lastYearSeedTrainX, percentageWinsTrainY)
print("Last season seed:")
print("Corr coeff:", correlationCoefficient)
print("P val:", pVal)

print("Testing the linear relationship between the features and the % wins")
correlationCoefficient, pVal = pearsonr(averageGameRatingSt5TrainX, percentageWinsTrainY)
print("Average in-game rating of top 5 players (usually starting 5):")
print("Corr coeff:", correlationCoefficient)
print("P val:", pVal)

print("Testing the linear relationship between the features and the % wins")
correlationCoefficient, pVal = pearsonr(inGameTeamRatingTrainX, percentageWinsTrainY)
print("In game rating of the team this season:")
print("Corr coeff:", correlationCoefficient)
print("P val:", pVal)

correlationCoefficient, pVal = pearsonr(numAllStarsTrainX, percentageWinsTrainY)
print("Number of all stars on the team this season:")
print("Corr coeff:", correlationCoefficient)
print("P val:", pVal)


# spearman's coefficient. 
print("\n\n\n")
print("Testing how well the relationship between two variables can be described using a monotonic function using spearman's corr coeff:")
print("Number of wins last season: ")
spearman_corr, spearman_p_value = spearmanr(lastYearNumWinsTrainX, percentageWinsTrainY)
print("Spearman Correlation Coefficient:", spearman_corr)
print("P-value:", spearman_p_value)

print("Last season seed: ")
spearman_corr, spearman_p_value = spearmanr(lastYearSeedTrainX, percentageWinsTrainY)
print("Spearman Correlation Coefficient:", spearman_corr)
print("P-value:", spearman_p_value)

print("Average in-game rating of top 5 players (usually starting 5):")
spearman_corr, spearman_p_value = spearmanr(averageGameRatingSt5TrainX, percentageWinsTrainY)
print("Spearman Correlation Coefficient:", spearman_corr)
print("P-value:", spearman_p_value)

print("In game rating of the team this season: ")
spearman_corr, spearman_p_value = spearmanr(inGameTeamRatingTrainX, percentageWinsTrainY)
print("Spearman Correlation Coefficient:", spearman_corr)
print("P-value:", spearman_p_value)

print("Number of all stars on the team this season: ")
spearman_corr, spearman_p_value = spearmanr(numAllStarsTrainX, percentageWinsTrainY)
print("Spearman Correlation Coefficient:", spearman_corr)
print("P-value:", spearman_p_value)





def draw3DGraph(master): #this is a function used by showDataVisualisation function. 
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    
    # Color map based on percentageWinsTrainY
    colors = cm.viridis(percentageWinsTrainY / max(percentageWinsTrainY))
    
    ax1.scatter(lastYearNumWinsTrainX, averageGameRatingSt5TrainX, percentageWinsTrainY, 
                c=colors, marker='o')

    ax1.set_xlabel('Number of wins last season')
    ax1.set_ylabel('Average in-game rating of top 5 players')
    ax1.set_zlabel('% Wins')
    ax1.set_title("3D visualisation of 2 key features and target variable.")

    # Add labels to each point of the team's name
    for i in range(len(nbaTeamsOrder)):
        ax1.text(lastYearNumWinsTrainX[i], averageGameRatingSt5TrainX[i], 
                 percentageWinsTrainY[i], teamNames[i], size=8, color='k')

    # Embed the 3D plot in the specified master frame
    canvas3D = FigureCanvasTkAgg(fig1, master=master)
    canvas3D.draw()
    canvas3D.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


## data visualisation

def showDataVisualisation(): # plotting each of the features with the output variable as well as an interactive 3d graph showing 2 key features (number of wins last season and average in-game rating of the team) and the output vairable. 
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


    featuresArrays = [lastYearNumWinsTrainX,lastYearSeedTrainX, averageGameRatingSt5TrainX, inGameTeamRatingTrainX, numAllStarsTrainX]
    xLabels = ["Num wins last season", "Seed last season", "Average in-game rating of top 5 players", "In game team rating this season", "Num all stars this season"]
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
        lastYearNumWinsTrainX, 
        averageGameRatingSt5TrainX, 
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
            lastYearNumWinsTrainX[i], 
            averageGameRatingSt5TrainX[i], 
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



## multiple linear regression: (this will not be as accurate as polynomial regression - curious to see how it performs)

A = np.vstack([lastYearNumWinsTrainX,lastYearSeedTrainX, averageGameRatingSt5TrainX, inGameTeamRatingTrainX, numAllStarsTrainX, np.ones(len(numAllStarsTrainX))]).T

# Calculate weights (coefficients) using least squares
results = np.linalg.lstsq(A, percentageWinsTrainY, rcond=None)
weightsList = results[0]
w1, w2, w3, w4, w5, c = weightsList
print(f"An increase in last season's number of wins will give the team {float(w1) * 100}% more wins this season as per my multiple linear regression regression model")
print(f"An increase of last season's seed by 1 position will give the team {float(w2) * 100}% more wins this season as per my multiple linear regression model")
print(f"An increase of this season's in-game rating by 1 point of the team will give the team {float(w3) * 100}% more wins this season as per my multiple linear regression model")
print(f"An increase of the average in-game rating of the team's top 5 players by 1 point will give the team {float(w4)}% more wins this season as per my multiple linear regression model")
print(f"One more all star will give the team {float(w4) * 100}% more wins this season as per my multiple linear regression model")


sumSquareResidualsLinReg = results[1]
print(f"MULTIPLE LINEAR REGRESSION: The sum of the square of residuals of the percentage of wins is {sumSquareResidualsLinReg * 100} in terms of % of wins in a season. ")




# calculating regression plane using least squares. 
# Create a grid for plotting the regression pl1ane
x1min, x1max = np.min(lastYearNumWinsTrainX), np.max(lastYearNumWinsTrainX)
x2min, x2max = np.min(lastYearSeedTrainX), np.max(lastYearSeedTrainX)
x3min, x3max = np.min(inGameTeamRatingTrainX), np.max(inGameTeamRatingTrainX)
x4min, x4max = np.min(averageGameRatingSt5TrainX), np.max(averageGameRatingSt5TrainX)
x5min, x5max = np.min(numAllStarsTrainX), np.max(numAllStarsTrainX)

# x1 = np.linspace(x1min, x1max, 40, dtype=np.float32)
# x2 = np.linspace(x2min, x2max, 40, dtype=np.float32)
# x3 = np.linspace(x3min, x3max, 40, dtype=np.float32)
# x4 = np.linspace(x4min, x4max, 40, dtype=np.float32)
# x5 = np.linspace(x5min, x5max, 40, dtype=np.float32)
# X1, X2, X3, X4, X5 = np.meshgrid(x1, x2, x3, x4, x5)


# # calculate the predicted y values (win percentage) from the regression coefficients
# Y = w1 * X1 + w2 * X2 + w3 * X3 + w4 * X4 + w5 * X5 + c
# # print(Y)



# #plotting the regression plane on the 3d graph with the 2 key features and the target vairable (this doesn't include the other 2 features as won't be able to plot - only included 2/4 features. )

# # let's plot all the data points: 
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # color map based on percentageWinsTrainY
# colors = cm.viridis(percentageWinsTrainY / max(percentageWinsTrainY))

# ax.scatter(lastYearNumWinsTrainX, inGameTeamRatingTrainX, percentageWinsTrainY, c=colors, marker='x')

# # plot the regression plane. 
# ax.plot_surface(X1, X2, Y, color='b', alpha=0.5)

# ax.set_xlabel('Number of wins last season')
# ax.set_ylabel("The team's in game rating this season")
# ax.set_zlabel('% wins')

# ax.set_title("Multiple linear regression line applied")

# ax.set_xlim(x1min, x1max)
# ax.set_ylim(x2min, x2max)

# # adding labels to each point
# for i in range(len(nbaTeamsOrder)):
#     ax.text(lastYearNumWinsTrainX[i], inGameTeamRatingTrainX[i], percentageWinsTrainY[i], teamNames[i], size=8, color='k')

# plt.show()



from sklearn.linear_model import LinearRegression


# now trying with polynomial regression: 
A = np.vstack([lastYearNumWinsTrainX,lastYearSeedTrainX, averageGameRatingSt5TrainX, inGameTeamRatingTrainX, numAllStarsTrainX]).T  
A = np.hstack([A, np.ones((A.shape[0], 1))])  

# print(A)

degree = 2
polynomial = PolynomialFeatures(2)
xPolynomial = polynomial.fit_transform(A)

polynomialRegressionModel = LinearRegression()
polynomialRegressionModel.fit(xPolynomial, percentageWinsTrainY)

polynomialWeights = polynomialRegressionModel.coef_
polynomialIntercept = polynomialRegressionModel.intercept_

print(f"These were the polynomial coefficients: {polynomialWeights}")
print(f"And this was the polynomial intercept: {polynomialIntercept}")

# x1min, x1max = np.min(lastYearNumWinsTrainX), np.max(lastYearNumWinsTrainX)
# x2min, x2max = np.min(lastYearSeedTrainX), np.max(lastYearSeedTrainX)
# x3min, x3max = np.min(inGameTeamRatingTrainX), np.max(inGameTeamRatingTrainX)
# x4min, x4max = np.min(averageGameRatingSt5TrainX), np.max(averageGameRatingSt5TrainX)
# x5min, x5max = np.min(numAllStarsTrainX), np.max(numAllStarsTrainX)

# no need to compute X1... X5 again, same as with multiple linear regression

# XGrid = np.column_stack((X1.ravel(), X2.ravel(), X3.ravel(), X4.ravel(), X5.ravel()))
# XGrid = np.hstack([XGrid, np.ones((XGrid.shape[0], 1))])  # Add bias term
# # print(XGrid)
# XGridPoly = polynomial.transform(XGrid)


# # calc the predicted plane / valyues from the polynomial model
# YPolynomialPredicted = polynomialRegressionModel.predict(XGridPoly).reshape(X1.shape)



# # Plotting the polynomial regression surface
# figPoly = plt.figure()
# axPoly = figPoly.add_subplot(111, projection='3d')

# # Create a color map based on percentageWinsTrainY
# colors_poly = cm.viridis(percentageWinsTrainY / max(percentageWinsTrainY))

# axPoly.scatter(numAllStarsTrainX, averageGameRatingSt5TrainX, percentageWinsTrainY, c=colors_poly, marker='o')

# # Plot the polynomial regression surface
# axPoly.plot_surface(X1, X2, YPolynomialPredicted, color='b', alpha=0.5)

# # Set labels
# axPoly.set_xlabel('Number of All Star Players')
# axPoly.set_ylabel('Average Rating of Top 5 Players')
# axPoly.set_zlabel('% Wins')

# axPoly.set_title("Polynomial Regression Surface")

# axPoly.set_xlim(x1min, x1max)
# axPoly.set_ylim(x2min, x2max)


# # Add labels to each point
# for i in range(len(nbaTeamsOrder)):
#     axPoly.text(numAllStarsTrainX[i], averageGameRatingSt5TrainX[i], percentageWinsTrainY[i], teamNames[i], size=8, color='k')

# plt.show()


# displaying how well of a fit the model is: 
#calculating sum of the square of residuals. 

inputData = np.vstack([lastYearNumWinsTrainX,lastYearSeedTrainX, averageGameRatingSt5TrainX, inGameTeamRatingTrainX, numAllStarsTrainX]).T
inputDataWithBias = np.hstack([inputData, np.ones((inputData.shape[0], 1))])
inputDataPoly = polynomial.transform(inputDataWithBias)
predictedValues = polynomialRegressionModel.predict(inputDataPoly)
residuals = percentageWinsTrainY - predictedValues
squaredResiduals = residuals ** 2
sumSquareResidualsPoly = np.sum(squaredResiduals)
print(f"The sum of the squared residuals on polynomial regression were: {sumSquareResidualsPoly * 100}% of wins whereas in multiple linear regression it was {sumSquareResidualsLinReg[0]*100}% of wins an improvement of {(float(sumSquareResidualsLinReg[0]) * 100) - (float(sumSquareResidualsPoly) * 100)}% of wins this season. ")



