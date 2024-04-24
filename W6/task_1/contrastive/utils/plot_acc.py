import matplotlib.pyplot as plt

acc = {'brush_hair': 0.4, 'catch': 0.36666666666666664, 'clap': 0.03333333333333333, 'climb_stairs': 0.1, 'draw_sword': 0.06666666666666667, 'drink': 0.16666666666666666, 'fall_floor': 0.06896551724137931, 'flic_flac': 0.06896551724137931, 'handstand': 0.0, 'hug': 0.5, 'kick': 0.0, 'kiss': 0.3448275862068966, 'pick': 0.0, 'pullup': 0.1, 'push': 0.16666666666666666, 'ride_bike': 0.6333333333333333, 'run': 0.13333333333333333, 'shoot_ball': 0.13793103448275862, 'shoot_gun': 0.0, 'situp': 0.2413793103448276, 'smoke': 0.034482758620689655, 'stand': 0.03333333333333333, 'sword': 0.06666666666666667, 'talk': 0.27586206896551724, 'turn': 0.13333333333333333, 'wave': 0.0, 'cartwheel': 0.06666666666666667, 'chew': 0.2, 'climb': 0.3, 'dive': 0.26666666666666666, 'dribble': 0.10344827586206896, 'eat': 0.2413793103448276, 'fencing': 0.0, 'golf': 0.9333333333333333, 'hit': 0.26666666666666666, 'jump': 0.16666666666666666, 'kick_ball': 0.2, 'laugh': 0.13333333333333333, 'pour': 0.4666666666666667, 'punch': 0.16666666666666666, 'pushup': 0.03333333333333333, 'ride_horse': 0.13333333333333333, 'shake_hands': 0.13793103448275862, 'shoot_bow': 0.1, 'sit': 0.06666666666666667, 'smile': 0.0, 'somersault': 0.3333333333333333, 'swing_baseball': 0.10344827586206896, 'sword_exercise': 0.0, 'throw': 0.1, 'walk': 0.13333333333333333}

plt.figure(figsize=(16, 4))
plt.bar(acc.keys(), acc.values(), color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Class Label')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy by Class')
plt.ylim(0, 1)
plt.show()