# Import dependencies
import numpy as np
import pandas as pd
import random
from math import comb
from collections import defaultdict
from matplotlib import pyplot as plt
from scipy.interpolate import griddata


def outputData(dataframe, filename):
    '''
    Outputs the given data as a csv file.
    '''
    dataframe.to_csv(filename, index=False)

# Generate Ballots, Profiles


def createAlternatives(num):
    '''
    Create list with desired number of alternatives. Example: createAlternatives(5) = ['0', '1', '2', '3', '4'].
    '''
    return [str(i) for i in range(num)]


def generateProfile(num_voters, alternatives):
    '''
    Given the number of voters and alternatives, this function returns a profile with each ballot a randomized shuffle of the alternatives.
    '''

    # Initialize empty list for profile
    profile = []

    for i in range(num_voters):
        # For every voter, randomize ranking of alternatives and add to profile
        ballot = random.sample(alternatives, len(alternatives))
        profile.append(ballot)

    return np.array(profile)


def getCondorcetWinner(profile):
    '''
    Finds the winner of a given profile (2D numpy array) using the Condorcet System: winner if head-to-head winner 
    against every other alternative. Also tracks the number of head-to-head ties.
    '''
    # Get num voters, num alternatives
    num_voters, num_alternatives = profile.shape

    # Initialize number of ties
    numTies = 0

    alternatives = [str(i) for i in range(num_alternatives)]
    winners = list()

    # Iterate through alternatives to compare to the other ones
    for alt in alternatives:

        # Create list and dictionary of the other choices in order to make comparisons and tracking scores easier
        others = list(set(alternatives) - set([alt]))
        others_scores = defaultdict(int)

        # Iterate through every ballot and compare to the other alternatives
        for diff_alt in others:
            for i in range(len(profile)):

                # index i < index j implies voter prefers i to j
                # if they prefer anything other than current choice, increment value associated with that alternative in dictionary
                if (list(profile[i]).index(alt) > list(profile[i]).index(diff_alt)):
                    others_scores[diff_alt] += 1

        for value in others_scores.values():
            if value == len(profile)/2:
                numTies += 1

        if (all(value <= len(profile)/2 for value in others_scores.values())):
            # If the other choices all got less or equal to half the vote, then current alternative is a Condorcet Winner
            winners.append(alt)

    return winners, numTies/2


def getCoombsWinner(profile):
    '''
    Finds the winner of a given profile (2D numpy array) using the Coombs System: iteratively remove most common
    last-place choice.
    '''
    # Get num voters, num alternatives
    num_voters, num_alternatives = profile.shape

    # Initialize dict to track last place votes
    scores = dict()
    for i in range(num_alternatives):
        scores[i] = 0

    eliminated = []

    while True:
        # Get last choice votes, add to score
        for voter in range(num_voters):
            ballot = profile[voter]
            last_choice_ind = -1
            while int(ballot[last_choice_ind]) in eliminated:
                last_choice_ind -= 1
                if last_choice_ind == -(num_alternatives):
                    break
            lastChoice = int(ballot[last_choice_ind])
            scores[lastChoice] = scores.get(lastChoice, 0) + 1

        # Eliminate alternative with most last-place votes
        maxReceived = max(scores.values())
        elim = [key for key in scores.keys() if scores[key] == maxReceived]

        for elimAlt in elim:
            del scores[elimAlt]

        eliminated += elim

        if len(eliminated) == num_alternatives:
            return [str(i) for i in elim]
        elif len(eliminated) == num_alternatives - 1:
            return [str(i) for i in [i for i in range(num_alternatives)] if i not in eliminated]


def getRCVWinner(profile):
    '''
    Finds the winner of a given profile (2D numpy array) using the Ranked-Choice Voting System: iteratively remove 
    least common first-place choice.
    '''
    # Get num voters, num alternatives
    num_voters, num_alternatives = profile.shape

    # Initialize dict to track last place votes
    scores = dict()
    for i in range(num_alternatives):
        scores[i] = 0

    eliminated = []

    while True:
        # Get first choice votes, add to score
        for voter in range(num_voters):
            ballot = profile[voter]
            first_choice_ind = 0
            while int(ballot[first_choice_ind]) in eliminated:
                first_choice_ind += 1
                if first_choice_ind == num_alternatives-1:
                    break
            firstChoice = int(ballot[first_choice_ind])
            scores[firstChoice] = scores.get(firstChoice, 0) + 1

        # Eliminate alternative with fewest votes
        minReceived = min(scores.values())
        elim = [key for key in scores.keys() if scores[key] == minReceived]

        for elimAlt in elim:
            del scores[elimAlt]

        eliminated += elim

        if len(eliminated) == num_alternatives:
            return [str(i) for i in elim]
        elif len(eliminated) == num_alternatives - 1:
            return [str(i) for i in [i for i in range(num_alternatives)] if i not in eliminated]


def runCondorcetFrequencySim(num_sims, num_alternatives, num_voters):
    '''
    Performs the simulation. Returns the frequency at which ties occur
    '''

    # Track
    numTies = 0
    numWinners = {}
    numComparisons = comb(num_alternatives, 2)

    # Set Dictionary to track how many times each number of winners occurs
    for i in range(num_sims):

        # Create alternatives, then profiles, then determine condorcet winner each time
        alternatives = createAlternatives(num_alternatives)
        profile = generateProfile(num_voters, alternatives)

        # Find winner(s)
        winners, num_ties = getCondorcetWinner(profile)

        num_winners = len(winners)

        numWinners[num_winners] = numWinners.get(num_winners, 0) + 1
        numTies += num_ties

    tie_frequency = (numTies/num_sims)/numComparisons

    return {key: 1.*value/num_sims for key, value in numWinners.items()}, tie_frequency


def runCoombsFrequencySim(num_sims, num_alternatives, num_voters):
    '''
    Performs the simulation. Returns the frequency at winners occur.
    '''
    # Track
    numWinners = {}

    # Set Dictionary to track how many times each number of winners occurs
    for i in range(num_sims):

        # Create alternatives, then profiles, then determine condorcet winner each time
        alternatives = createAlternatives(num_alternatives)
        profile = generateProfile(num_voters, alternatives)

        # Find winner(s)
        winners = getCoombsWinner(profile)
        num_winners = len(winners)

        numWinners[num_winners] = numWinners.get(num_winners, 0) + 1

    return {key: 1.*value/num_sims for key, value in numWinners.items()}


def runRCVvsCoombsFrequencySim(num_sims, num_alternatives, num_voters):
    '''
    Performs the simulation. Returns the frequency at which winners match from Coombs to RCV.
    '''
    num_identical_winners = 0
    num_same_num_winners = 0

    # Set Dictionary to track how many times each number of winners occurs
    for i in range(num_sims):

        # Create alternatives, then profiles, then determine condorcet winner each time
        alternatives = createAlternatives(num_alternatives)
        profile = generateProfile(num_voters, alternatives)

        # Find winner(s)
        rcv_winners = getRCVWinner(profile)
        coombs_winners = getCoombsWinner(profile)

        if set(rcv_winners) == set(coombs_winners):
            num_identical_winners += 1

        if len(rcv_winners) == len(coombs_winners):
            num_same_num_winners += 1

    return num_identical_winners/num_sims, num_same_num_winners/num_sims


# House simulation
num_sims = 10000
num_alts_max = 5
num_voters_min = 5
num_voters_max = 100

data = []
columns = ["num_voters", "num_alternatives",
           "identical_winners_frequency", "same_number_of_winners_frequency"]
data.append(columns)

# Run simulation for every pair of num_alternatives and num_voters in the given range
for num_alternatives in range(2, num_alts_max+1):
    for num_voters in range(num_voters_min, num_voters_max+1):
        print(num_voters, num_alternatives)

        # Get results
        identical_rate, same_num_rate = runRCVvsCoombsFrequencySim(
            num_sims, num_alternatives, num_voters)

        # Initialize list for data to be entered (eventually will be appended to data)
        row = []

        # Enter results into row
        row.append(num_voters)
        row.append(num_alternatives)
        row.append(identical_rate)
        row.append(same_num_rate)
        data.append(row)

df = pd.DataFrame(data[1:], columns=data[0])
outputData(df, "RCV_coombs_data.csv")
