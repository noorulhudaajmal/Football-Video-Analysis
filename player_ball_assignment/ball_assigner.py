from utils import get_bbox_center, measure_distance

 
class BallAssigner:
    """
    Assigns the ball to the closest 
    player based on bounding box proximity.
    """

    def __init__(self):
        """
        Initializes the ball assigner 
        with a maximum distance threshold.
        """

        self.max_distance = 70
        
    
    def assign_ball_to_player(self, players, ball_bbox):
        """
        Assigns the ball to the closest player based on 
        the distance between the ball and player bounding boxes.
        """

        ball_center = get_bbox_center(bbox=ball_bbox)
        
        min_dist = 99999
        closest_player = -1
        
        for player_id, player in players.items():
            player_bbox = player["bbox"]
            
            dist_left = measure_distance(bbox_pt1=(player_bbox[0], player_bbox[-1]), bbox_pt2=ball_center)
            dist_right = measure_distance(bbox_pt1=(player_bbox[2], player_bbox[-1]), bbox_pt2=ball_center)
            
            dist = min(dist_left, dist_right)
            
            if dist < self.max_distance:
                if dist < min_dist:
                    min_dist = dist
                    closest_player = player_id
                    
        return closest_player