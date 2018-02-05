# -*- encoding:utf-8 -*-

import os
import random
import flags as fl
import numpy as np

class UserModel(object):
    def __init__(self, filePath, batchSize):
        self.filePath = filePath
        self.batchSize = batchSize
        self.currentIndex = 0
        self.dataList = []
        self.dataMaps = {}
        self.trainDataLength = 0

    def nextEpoch(self):
        self.currentIndex = 0

    def randomList(self):
        random.shuffle(self.dataList)

    def getDataListLength(self):
        return len(self.dataList)

    def setTrainDataLength(self, length):
        self.trainDataLength = length

    def next(self):
        self.currentIndex = self.currentIndex + self.batchSize

    def hasNext(self):
        if self.currentIndex >= self.trainDataLength:
            return False
        return True

    def getNext(self, tag, maxLength=0, isBinary=True):
        if (self.currentIndex + self.batchSize) > self.trainDataLength:
            return None
        tagList = []
        for i in range(self.batchSize):
            tagList.append(self.dataList[i + self.currentIndex][tag])
        if isBinary:
            return self.toBinaryList(tagList, maxLength)
        return tagList

    def getNextByIds(self, idLists, tag, maxLength=0, isBinary=True):
        tagList = []
        for key in idLists:
            if not self.dataMaps.has_key(key):
                return None
            if not self.dataMaps[key].has_key(tag):
                return None
            tagList.append(self.dataMaps[key][tag])
        if isBinary:
            return self.toBinaryList(tagList, maxLength)
        return tagList

    def getTestData(self, tag, maxLength, isBinary=True):
        if self.trainDataLength == 0 or self.trainDataLength >= len(self.dataList):
            return None
        tagList = []
        testDataLength = len(self.dataList) - self.trainDataLength
        for i in range(testDataLength):
            tagList.append(self.dataList[i + self.trainDataLength][tag])
        if isBinary:
            return self.toBinaryList(tagList, maxLength)
        return tagList

    def toBinaryList(self, dataList, maxLength):
        binaryList = []
        for i in dataList:
            binary = [int(j) for j in list('{0:0b}'.format(i))]
            if len(binary) != maxLength:
                left = maxLength - len(binary)
                while left != 0:
                    binary.insert(0, 0)
                    left = left - 1
            binaryList.extend(binary)
        return np.reshape(binaryList, [len(dataList), maxLength])

    def toIntegerList(self, dataList, maxLength):
        integerList = []
        for i in dataList:
            if type(i) == int:
                integerList.append(i)
            else:
                integerList.append(ord(i))

        leftSize = maxLength - len(dataList)
        while leftSize != 0:
            integerList.append(0)
            leftSize = leftSize - 1
        return integerList

class UserManager(UserModel):
    def __init__(self, filePath, batchSize):
        UserModel.__init__(self, filePath, batchSize)
        self.dataMaps = {}
        self.occupations = (
            'other', 'academic/educator', 'artist', 'clerical/admin',
            'college/grad student', 'customer service', 'doctor/health care',
            'executive/managerial', 'farmer', 'homemaker', 'K-12 student', 'lawyer',
            'programmer', 'retired', 'sales/marketing', 'scientist', 'self-employed',
            'technician/engineer', 'tradesman/craftsman', 'unemployed', 'writer'
        )
        self.readConfig()
        self.randomList()

    def transforAge(self, age):
        if age <= 10:
            return 0
        elif age < 19:
            return 1
        elif age < 25:
            return 2
        elif age < 32:
            return 3
        elif age < 40:
            return 4
        elif age < 50:
            return 5
        else:
            return 6

    def getOccupation(self, occupation):
        if occupation > len(self.occupations):
            return 0
        return self.occupations[occupation]

    def readConfig(self):
        with open(self.filePath) as file:
            lines = file.readlines()
            for line in lines:
                if len(line) == 0:
                    continue
                infos = line.split('::')
                info = {
                    'id': int(infos[0]),
                    'sex': 0 if infos[1] == 'F' else 1,
                    'age': int(self.transforAge(infos[2])),
                    'occ': int(infos[3])
                }
                self.dataMaps[info['id']] = info
                self.dataList.append(info)

    def getUserInfo(self, userId):
        if not self.dataMaps.has_key(userId):
            return None
        return self.dataMaps[userId]

    def getAllUserInfos(self):
        infos = []
        for (userId, info) in self.dataMaps.items():
            infos.append(info)
        return infos

    def getNextUID(self):
        return self.getNext('id', fl.uid_dimension)

    def getNextUserSex(self):
        return self.getNext('sex', fl.sex_dimension)

    def getNextUserAge(self):
        return self.getNext('age', fl.sex_dimension)

    def getNextUserJob(self):
        return self.getNext('occ', fl.occ_dimension)
    
    def getTestUID(self):
        return self.getTestData('id', fl.uid_dimension)

    def getTestUserSex(self):
        return self.getTestData('sex', fl.sex_dimension)

    def getTestUserAge(self):
        return self.getTestData('age', fl.sex_dimension)

    def getTestUserJob(self):
        return self.getTestData('occ', fl.occ_dimension)

    def getUserIds(self, idList):
        return self.getNextByIds(idList, 'id', fl.uid_dimension)

    def getUserSexs(self, idList):
        return self.getNextByIds(idList, 'sex', fl.sex_dimension)

    def getUserAges(self, idList):
        return self.getNextByIds(idList, 'age', fl.age_dimension)

    def getUserJobs(self, idList):
        return self.getNextByIds(idList, 'occ', fl.occ_dimension)

class MovieManager(UserModel):
    def __init__(self, filePath, batchSize):
        UserModel.__init__(self, filePath, batchSize)
        self.moveGenres = {
            "Action": 0,
            "Adventure": 1,
            "Animation": 2,
            "Children's": 3,
            "Comedy": 4,
            "Crime": 5,
            "Documentary": 6,
            "Drama": 7,
            "Fantasy": 8,
            "Film-Noir": 9,
            "Horror": 10,
            "Musical": 11,
            "Mystery": 12,
            "Romance": 13,
            "Sci-Fi": 14,
            "Thriller": 15,
            "War": 16,
            "Western": 17,
        }
        self.readConfig()
        self.randomList()

    def transforGenres(self, genres):
        genresNumber = 0
        infos = genres.split('|')
        for info in infos:
            if info.endswith('\n'):
                info = info[:-1]
            if not self.moveGenres.has_key(info):
                print 'error:', info
                return 0
            genresNumber = genresNumber | (2 ** self.moveGenres[info])
        return genresNumber

    def translateAllWords(self):
        words = []
        allWords = {}
        for line in self.dataList:
            lineArrs = line['name'].split(' ')
            for word in lineArrs:
                if word.startswith('(') and word.endswith(')'):
                    continue
                allWords[word] = True
        index = 0
        for (word, _) in allWords.items():
            allWords[word] = index
            index = index + 1
        for line in self.dataList:
            currentWordList = []
            lineArrs = line['name'].split(' ')
            for word in lineArrs:
                if word.startswith('(') and word.endswith(')'):
                    continue
                currentWordList.append(allWords[word])
            line['name'] = currentWordList

    def readConfig(self):
        with open(self.filePath) as file:
            lines = file.readlines()
            for line in lines:
                if len(line) == 0:
                    continue
                infos = line.split('::')
                info = {
                    'id': int(infos[0]),
                    'name': infos[1],
                    'genres': self.transforGenres(infos[2]),
                }
                self.dataMaps[info['id']] = info
                self.dataList.append(info)
            self.translateAllWords()

    def getMovieInfo(self, movieId):
        if not self.dataMaps.has_key(movieId):
            return None
        return self.dataMaps[movieId]

    def getAllMovieInfos(self):
        infos = []
        for (userId, info) in self.dataMaps.items():
            infos.append(info)
        return infos

    def getNextMID(self):
        return self.getNext('id', fl.mid_dimension)

    def getNextName(self):
        return self.getNext('name', isBinary=False)

    def getNextGenres(self):
        return self.getNext('genres', fl.category_dimension)

    def getTestMID(self):
        return self.getTestData('id', fl.mid_dimension)

    def getTestName(self):
        return self.getTestData('name', isBinary=False)

    def getTestGenres(self):
        return self.getTestData('genres', fl.category_dimension)

    def getMovieIds(self, idList):
        return self.getNextByIds(idList, 'id', fl.mid_dimension)

    def getMovieGenres(self, idList):
        return self.getNextByIds(idList, 'genres', fl.category_dimension)

    def getMovieNames(self, idList):
        movieNameList = self.getNextByIds(idList, 'name', isBinary=False)
        movieNameIntegerList = []
        for name in movieNameList:
            movieNameIntegerList.append(self.toIntegerList(name, 15))
        return np.reshape(movieNameIntegerList, [len(movieNameIntegerList), len(movieNameIntegerList[0])])

class RatingManager(UserModel):
    def __init__(self, filePath, batchSize):
        UserModel.__init__(self, filePath, batchSize)
        self.readConfig()
        self.randomList()

    def readConfig(self):
        with open(self.filePath) as file:
            lines = file.readlines()
            for line in lines:
                if len(line) == 0:
                    continue
                infos = line.split('::')
                info = {
                    'userId': int(infos[0]),
                    'movieId': int(infos[1]),
                    'rating': int(infos[2]),
                    'timestamp': int(infos[3][:-1]),
                }
                if not self.dataMaps.has_key(infos[0]):
                    self.dataMaps[infos[0]] = {}
                self.dataMaps[infos[0]][infos[1]] = {
                    'rating': int(infos[2]),
                    'timestamp': int(infos[3]),
                }
                self.dataList.append(info)

    def getRatingInfo(self, userId, movieId):
        if not self.dataMaps.has_key(userId):
            return None
        if not self.dataMaps[userId].has_key(movieId):
            return None
        return self.dataMaps[userId][movieId]

    def getRatingInfoByIndex(self, index):
        if index > len(self.dataList):
            return None
        return self.dataList[index - 1]

    def getAllRatingInfos(self):
        return self.dataList

    def getNextUID(self):
        return self.getNext('userId', isBinary=False)

    def getNextMID(self):
        return self.getNext('movieId', isBinary=False)

    def getNextRating(self):
        return np.reshape(self.getNext('rating', isBinary=False), [self.batchSize, 1])

    def getNextTimestamp(self):
        return self.getNext('timestamp')

    def getTestUID(self):
        return self.getTestData('userId', fl.uid_dimension, isBinary=False)

    def getTestMID(self):
        return self.getTestData('movieId', fl.mid_dimension, isBinary=False)

    def getTestRating(self):
        return np.reshape(self.getTestData('rating', 1, isBinary=False), [-1, 1])

    def getTestTimestamp(self):
        return self.getTestData('timestamp')

class InputManager(object):
    def __init__(self, batchSize):
        self.userManager = UserManager('/Users/sjjwind/Downloads/ml-1m/users.dat', batchSize)
        self.movieManager = MovieManager('/Users/sjjwind/Downloads/ml-1m/movies.dat', batchSize)
        self.ratingManager = RatingManager('/Users/sjjwind/Downloads/ml-1m/ratings.dat', batchSize)
        
        userListLength = self.userManager.getDataListLength()
        movieListLength = self.movieManager.getDataListLength()
        ratingListLength = self.ratingManager.getDataListLength()

        # 将数据按照9:1的方式，将数据分为训练样本与测试样本
        userTrainLength = int(int(userListLength * 0.9) / batchSize) * batchSize
        movieTrainLength = int(int(movieListLength * 0.9) / batchSize) * batchSize
        ratingTrainLength = int(int(ratingListLength * 0.9) / batchSize) * batchSize

        self.userManager.setTrainDataLength(userTrainLength)
        self.movieManager.setTrainDataLength(movieTrainLength)
        self.ratingManager.setTrainDataLength(ratingTrainLength)

    def nextEpoch(self):
        self.ratingManager.nextEpoch()

    def next(self):
        self.ratingManager.next()

    def hasNext(self):
        return self.ratingManager.hasNext()

    def getNextUserInfos(self):
        uidLists = self.ratingManager.getNextUID()
        userIds = self.userManager.getUserIds(uidLists)
        userSexs = self.userManager.getUserSexs(uidLists)
        userAges = self.userManager.getUserAges(uidLists)
        userJobs = self.userManager.getUserJobs(uidLists)
        return userIds, userSexs, userAges, userJobs

    def getNextMovieInfos(self):
        midLists = self.ratingManager.getNextMID()
        movieIds = self.movieManager.getMovieIds(midLists)
        movieNames = self.movieManager.getMovieNames(midLists)
        movieGenres = self.movieManager.getMovieGenres(midLists)
        return movieIds, movieGenres, movieNames

    def getNextTargets(self):
        return self.ratingManager.getNextRating()

    def getTestUserInfos(self):
        uidLists = self.ratingManager.getTestUID()
        userIds = self.userManager.getUserIds(uidLists)
        userSexs = self.userManager.getUserSexs(uidLists)
        userAges = self.userManager.getUserAges(uidLists)
        userJobs = self.userManager.getUserJobs(uidLists)
        return userIds, userSexs, userAges, userJobs

    def getTestMovieInfos(self):
        midLists = self.ratingManager.getTestMID()
        movieIds = self.movieManager.getMovieIds(midLists)
        movieNames = self.movieManager.getMovieNames(midLists)
        movieGenres = self.movieManager.getMovieGenres(midLists)
        return movieIds, movieGenres, movieNames

    def getTestTargets(self):
        return self.ratingManager.getTestRating()

    def getUserInput(self):
        return self.userManager

    def getMovieInput(self):
        return self.movieManager

    def getRatingInput(self):
        return self.ratingManager

def test():
    # userManager = UserManager('/Users/sjjwind/Downloads/ml-1m/users.dat')
    # print userManager.getUserInfo(100)

    # movieManager = MovieManager('/Users/sjjwind/Downloads/ml-1m/movies.dat')
    # print movieManager.getMovieInfo(90)

    # ratingManager = RatingManager('/Users/sjjwind/Downloads/ml-1m/dataList.dat')
    # ratingInfo = ratingManager.getRatingInfoByIndex(100)
    # print ratingInfo
    # print userManager.getUserInfo(ratingInfo['userId'])
    # print movieManager.getMovieInfo(ratingInfo['movieId'])

    inputManager = InputManager(96)
    print '#' * 30
    print inputManager.getUserInput().getNextUID()
    print inputManager.getUserInput().getNextUserAge()
    print inputManager.getUserInput().getNextUserJob()
    print inputManager.getUserInput().getNextUserSex()
    print '#' * 30
    print inputManager.getMovieInput().getNextMID()
    print inputManager.getMovieInput().getNextName()
    print inputManager.getMovieInput().getNextGenres()

if __name__ == '__main__':
    test()
