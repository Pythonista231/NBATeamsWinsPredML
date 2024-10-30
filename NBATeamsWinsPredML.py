import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from statistics import mean 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from sklearn.preprocessing import PolynomialFeatures




# predicting the number of wins a team will have based on the number of all starts they have as well as the average game rating of their starting 5. 
# using multiple linear regression at first. 

nbaTeamsOrder = np.array([ "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls", 
                  "Cleveland Cavaliers", "Detroit Pistons", "Indiana Pacers", "Miami Heat", "Milwaukee Bucks", 
                  "New York Knicks", "Orlando Magic", "Philadelphia 76ers", "Toronto Raptors", "Washington Wizards", 
                  "Dallas Mavericks", "Denver Nuggets", "Golden State Warriors", "Houston Rockets", "Los Angeles Clippers", 
                  "Los Angeles Lakers", "Memphis Grizzlies", "Minnesota Timberwolves", "New Orleans Pelicans", 
                  "Oklahoma City Thunder", "Phoenix Suns", "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Utah Jazz",])
teamNames = np.array([teamName.split()[-1] for teamName in nbaTeamsOrder])

numAllStarsTrainX = np.array([0, 3, 0, 0, 1, 1, 0, 2, 1, 2, 1, 0, 1, 0, 0, 2, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 0, 2, 0, 1,])


# extracting the average game rating of the team's starting 5. for this, I am going to use the NBA 2k25 game since they have player ratings. 
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
averageGameRatingSt5TrainX = np.array([81.6, 89.4, 79.2, 81.2, 80.6, 84.2, 80.6, 84.6, 83.8, 86.2, 86.2, 83, 86.2, 81.2, 80.4, 86.4, 85.6, 83.2, 82.6, 82.8, 86, 83.6, 85, 85, 86.4, 86, 81.2, 84.6, 82, 80.6]) #hard coded it because next year the url i used to web scrape will show next year's data so won't be compatible with rest of training data. 

percentageWinsTrainY = np.array([
    0.434, 0.785, 0.390, 0.256, 0.476, 0.564, 0.171, 0.567, 0.539, 0.580,
    0.600, 0.562, 0.562, 0.305, 0.183, 0.621, 0.681, 0.554, 0.500, 0.602,
    0.562, 0.329, 0.681, 0.568, 0.685, 0.570, 0.256, 0.560, 0.268, 0.378
])


# let's plot all the data points: 
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')

# color map based on percentageWinsTrainY
colors = cm.viridis(percentageWinsTrainY / max(percentageWinsTrainY))

ax1.scatter(numAllStarsTrainX,averageGameRatingSt5TrainX, percentageWinsTrainY, c=colors, marker='o')

ax1.set_xlabel('Number of All Star Players')
ax1.set_ylabel('Average Rating of Top 5 Players')
ax1.set_zlabel('% Wins')

ax1.set_title("Data visualisation before machine learning")

# Add labels to each point of the team's name. 
for i in range(len(nbaTeamsOrder)):
    ax1.text(numAllStarsTrainX[i], averageGameRatingSt5TrainX[i], percentageWinsTrainY[i], teamNames[i], size=8, color='k')

plt.show()


### now let's try doing multiple linear regression and see how it goes: 

A = np.vstack([numAllStarsTrainX, averageGameRatingSt5TrainX, np.ones(len(numAllStarsTrainX))]).T

# Calculate weights (coefficients) using least squares
results = np.linalg.lstsq(A, percentageWinsTrainY, rcond=None)
weightsList = results[0]
w1, w2, c = weightsList
print(f"An increase in one all star give the team {float(w1) * 100}% more wins, and if a team gets a 1 point higher average rating of their top 5 players it means they will win {float(w2) * 100}% more games in a season. There is also a bias of {c}")

sumOfSquareOfResiduals = results[1]
print(f"The sum of the square of residuals of the percentage of wins is {sumOfSquareOfResiduals * 100} in terms of % of wins in a season. ")




# calculating regression plane using least squares. 
# Create a grid for plotting the regression pl1ane
x1min, x1max = np.min(numAllStarsTrainX), np.max(numAllStarsTrainX)
x2min, x2max = np.min(averageGameRatingSt5TrainX), np.max(averageGameRatingSt5TrainX)

x1 = np.linspace(x1min, x1max, 100)
# print(x1)
x2 = np.linspace(x2min, x2max, 100)
# print(x2)
X1, X2 = np.meshgrid(x1, x2)

# calculate the predicted y values (win percentage) from the regression coefficients
Y = w1 * X1 + w2 * X2 + c
# print(Y)
# print(Y)


#plotting the regression plane. 

# let's plot all the data points: 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# color map based on percentageWinsTrainY
colors = cm.viridis(percentageWinsTrainY / max(percentageWinsTrainY))

ax.scatter(numAllStarsTrainX, averageGameRatingSt5TrainX, percentageWinsTrainY, c=colors, marker='x')

# plot the regression plane. 
ax.plot_surface(X1, X2, Y, color='b', alpha=0.5)

ax.set_xlabel('Number of All Star Players')
ax.set_ylabel('Average Rating of Top 5 Players')
ax.set_zlabel('% Wins')

ax.set_title("Multiple linear regression line applied")

ax.set_xlim(x1min, x1max)
ax.set_ylim(x2min, x2max)

# adding labels to each point
for i in range(len(nbaTeamsOrder)):
    ax.text(numAllStarsTrainX[i], averageGameRatingSt5TrainX[i], percentageWinsTrainY[i], teamNames[i], size=8, color='k')

plt.show()


from sklearn.linear_model import LinearRegression


# now trying with polynomial regression: 
A = np.vstack([numAllStarsTrainX, averageGameRatingSt5TrainX]).T  
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

x1min, x1max = np.min(numAllStarsTrainX), np.max(numAllStarsTrainX)
x2min, x2max = np.min(averageGameRatingSt5TrainX), np.max(averageGameRatingSt5TrainX)

x1 = np.linspace(x1min, x1max, 100)
x2 = np.linspace(x2min, x2max, 100)
X1, X2 = np.meshgrid(x1, x2)

XGrid = np.column_stack((X1.ravel(), X2.ravel()))
XGrid = np.hstack([XGrid, np.ones((XGrid.shape[0], 1))])  # Add bias term
# print(XGrid)
XGridPoly = polynomial.transform(XGrid)


# calc the predicted plane / valyues from the polynomial model
YPolynomialPredicted = polynomialRegressionModel.predict(XGridPoly).reshape(X1.shape)



# Plotting the polynomial regression surface
figPoly = plt.figure()
axPoly = figPoly.add_subplot(111, projection='3d')

# Create a color map based on percentageWinsTrainY
colors_poly = cm.viridis(percentageWinsTrainY / max(percentageWinsTrainY))

axPoly.scatter(numAllStarsTrainX, averageGameRatingSt5TrainX, percentageWinsTrainY, c=colors_poly, marker='o')

# Plot the polynomial regression surface
axPoly.plot_surface(X1, X2, YPolynomialPredicted, color='b', alpha=0.5)

# Set labels
axPoly.set_xlabel('Number of All Star Players')
axPoly.set_ylabel('Average Rating of Top 5 Players')
axPoly.set_zlabel('% Wins')

axPoly.set_title("Polynomial Regression Surface")

axPoly.set_xlim(x1min, x1max)
axPoly.set_ylim(x2min, x2max)


# Add labels to each point
for i in range(len(nbaTeamsOrder)):
    axPoly.text(numAllStarsTrainX[i], averageGameRatingSt5TrainX[i], percentageWinsTrainY[i], teamNames[i], size=8, color='k')

plt.show()


# displaying how well of a fit the model is: 
#calculating square of sum of residuals. 

inputData = np.vstack([numAllStarsTrainX, averageGameRatingSt5TrainX]).T
inputDataWithBias = np.hstack([inputData, np.ones((inputData.shape[0], 1))])
inputDataPoly = polynomial.transform(inputDataWithBias)
predictedValues = polynomialRegressionModel.predict(inputDataPoly)
residuals = percentageWinsTrainY - predictedValues
squaredResiduals = residuals ** 2
sumOfSquaredResiduals = np.sum(squaredResiduals)
print(f"The sum of the squared residuals on polynomial regression were: {sumOfSquaredResiduals} whereas in multiple linear regression it was {sumOfSquareOfResiduals[0]} a ~3% difference in wins per season")
