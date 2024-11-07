import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class TeamAssigner:
    """
    Assigns players to teams based on shirt color using clustering.
    """

    def __init__(self):
        """
        Initializes the team assigner with dictionaries 
        for team colors and player-team assignments.
        """

        self.team_colors = {}
        self.player_team_dict = {}


    def get_clustering_model(self, img):
        """
        Generates a KMeans clustering model to 
        identify dominant shirt colors in an image.
        """

        img = img.reshape(-1,3)
        
        kmeans = KMeans(n_clusters=2, init="k-means++", random_state=101, n_init=1)
        kmeans.fit(img)
        
        return kmeans
    
    
    def get_player_shirt_color(self, frame, bbox):
        """
        Extracts and clusters the shirt color of a 
        player from the given frame and bounding box.
        """

        player_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        player_shirt = player_img[0: int(player_img.shape[0]/2), :]
        
        kmeans = self.get_clustering_model(player_shirt)
        cluster_labels = kmeans.labels_
        
        clustered_img = cluster_labels.reshape((player_shirt.shape[0], player_shirt.shape[1]))
        
        corner_clusters = [clustered_img[0,0], clustered_img[0,-1], clustered_img[-1,0], clustered_img[-1,-1]]
        bg_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1-bg_cluster
        
        shirt_color = kmeans.cluster_centers_[player_cluster]
        
        return shirt_color
        
        
    def assign_team_color(self, frame, player_detections):
        """
        Assigns each player to one of two teams 
        based on the clustering of shirt colors.
        """
        
        shirt_colors = []
        
        for _, player in player_detections.items():
            bbox = player["bbox"]
            player_shirt_color = self.get_player_shirt_color(frame, bbox)
            
            shirt_colors.append(player_shirt_color)
            
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=101)
        kmeans.fit(shirt_colors)
        
        self.kmeans = kmeans
        
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
        
            
        return shirt_colors

    
    def get_player_team(self, frame, player_bbox, player_id):
        """
        Returns the team ID of a player based on 
        their shirt color, using kmeans clustering algorithm.
        """

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_shirt_color = self.get_player_shirt_color(frame=frame, bbox=player_bbox)
        
        team_id = self.kmeans.predict(player_shirt_color.reshape(1,-1))[0]
        team_id+=1
        
        self.player_team_dict[player_id] = team_id
        
        return team_id
    
    