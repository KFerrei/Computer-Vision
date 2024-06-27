import cv2
def import_img(name_ref, name_que, equalize = False):
  image_ref = cv2.imread(name_ref + "001.jpg", 0)
  image_que = cv2.imread(name_que + "001.jpg", 0)
  if equalize:
    image_ref = cv2.equalizeHist(image_ref)
    image_que = cv2.equalizeHist(image_que)
  images_ref = [(1, image_ref)]
  images_que = [(1, image_que)]

  i = 2
  while image_que is not None:
    n_image =  "{:03d}".format(i) + ".jpg"
    path_que = name_que + n_image
    image_que = cv2.imread(path_que, 0)
    if image_que is not None:
      if equalize:
        image_que = cv2.equalizeHist(image_que)
      images_que.append((i,image_que))
    i +=1

  i=2
  while image_ref is not None:
    n_image =  "{:03d}".format(i) + ".jpg"
    path_ref = name_ref + n_image
    image_ref = cv2.imread(path_ref, 0)
    if image_ref is not None:
      if equalize:
        image_ref = cv2.equalizeHist(image_ref)
      images_ref.append((i,image_ref))
    i +=1
  return images_ref, images_que

def get_param(images_que, images_ref, n_features= 500, scale_factor = 1.2, n_levels = 8, edge_Threshold=8):
  orb = cv2.ORB_create(n_features, scale_factor, n_levels, edge_Threshold)
  par_query = {}
  par_references = {}
  for i in range(1, len(images_ref)+1):
    img = images_ref[i-1][1]
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    par_references[images_ref[i-1][0]] = (kp, des)
  for i in range(1, len(images_que)+1):
    img = images_que[i-1][1]
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    par_query[images_que[i-1][0]] = (kp, des)
  return par_references, par_query

def get_good(des1, des2, ratio = 0.8):
  # create BFMatcher object, see (3) above
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)

  # Match descriptors, see (3) above
  matches = bf.knnMatch(des1, des2, k=2)
  # Apply ratio test, see (3) above
  good = []
  for m, n in matches:
      if m.distance < ratio * n.distance:
          good.append(m)
  return good

def matching_scores(query_image, images_ref, par_references, par_query, ratio = 0.8):
  score = {}
  for j in range(1, len(images_ref)+1):
    img = images_ref[j-1][1]
    (kp1, des1) = par_query
    (kp2, des2) = par_references[j] 
    good = get_good(des1, des2, ratio)
    # Create src_pts and dst_pts as float arrays to be passed into cv2.,findHomography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    # using RANSAC
    if len(src_pts) < 4:
      if 0 in score:
        score[0].append(j)
      else:
        score[0] = [j]
    else:
      M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
      mask = mask.ravel().tolist()
      score_j = sum(mask)
      if score_j in score:
        score[score_j].append(j)
      else:
        score[score_j] = [j]
  return score

def identify_image(query_image, images_ref, par_references, par_query, k=1, thr = 5, ratio = 0.8):
  score = matching_scores(query_image, images_ref, par_references, par_query, ratio)
  score_ranked = sorted(score, reverse=True)
  best_scores = []
  for i in range(k):
    if len(score_ranked)>i:
      scr = score_ranked[i]
      if scr>= thr:
        for j in score[scr]:
          best_scores.append((j,scr))
  return best_scores

def identify_images(images_que, images_ref, par_references, par_query, k=1, thr = 5, ratio = 0.8):
  best_scores = {}
  for i in range(1, len(images_que)+1):
    query_image = images_que[i-1]
    best_score = identify_image(query_image, images_ref, par_references, par_query[images_que[i-1][0]], k, thr, ratio)
    best_scores[images_que[i-1][0]] = best_score
  return best_scores

# Your code to identify query objects and measure search accuracy for data set here 
def identify_image_top_k(query_image, images_ref, k, thr = 5):
  score = matching_scores(query_image, images_ref)
  score_ranked = sorted(score, reverse=True)
  best_scores = []
  print(score_ranked)
  for i in range(k):
    if score_ranked[i]>= thr:
      best_scores.append([i for i, x in enumerate(score) if x == score_ranked[i]])
  return best_scores