import os
import sys
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

def compute_mse(predict, original):
    err = predict - original 
    return np.mean(err ** 2)


def predict_frame(refFrame, motionVectors, blockSize):
    """
       predict a frame given the reference frame and motion vectors
    """
    # Initialize output frame with same size and type as reference frame
    curFrame = np.zeros(refFrame.shape, dtype=refFrame.dtype)

    # Loop through all blocks in the image
    for y in range(0, curFrame.shape[0], blockSize):
        for x in range(0, curFrame.shape[1], blockSize):
            # Get the motion vector for this block
            motion = motionVectors[y // blockSize, x // blockSize]

            # Calculate the coordinates of the top-left corner of the block in the reference frame
            refX = x + int(motion[0]) # here we directly convert float to int 
            refY = y + int(motion[1])

            # Make sure the block is within the bounds of the reference frame
            if refX >= 0 and refX + blockSize <= refFrame.shape[1] and \
               refY >= 0 and refY + blockSize <= refFrame.shape[0]:
                # Copy the block from the reference frame to the current frame at the new location
                block = refFrame[refY: refY+blockSize, refX: refX+blockSize]
                curFrame[y:y+blockSize, x:x+blockSize] = block

    return curFrame


def visualize_motion_vectors(refFrame, motionVectors, blockSize):
    motionVectorImg = np.zeros((refFrame.shape[0], refFrame.shape[1], 3), dtype=np.uint8)
    
    for i in range(0, refFrame.shape[0], blockSize):
        for j in range(0, refFrame.shape[1], blockSize):
            mv = motionVectors[i//blockSize, j//blockSize]
            arrowLength = mv[0] * mv[0] + mv[1] * mv[1]
            arrowLength = np.sqrt(arrowLength)
            cv2.arrowedLine(motionVectorImg, (j, i), (j + int(mv[1]), i + int(mv[0])), (0, 255, 0), 1)
    
    return motionVectorImg



def motion_estimation_full_search(refFrame, curFrame, blockSize, searchRange):
    """
     motion estimation by full search 
     @refFrame: H*W, reference frame with single channel
     @curFrame: H*W, current frame with single channel
     @blockSize: size of block, typically set to 16
     @searchRange: search range [-p, p], typically p is set 15 or 7
    """
    rows, cols = refFrame.shape[:2]
    mv_rows = rows // blockSize
    mv_cols = cols // blockSize
    motionVectors = np.zeros((mv_rows, mv_cols, 2), dtype=np.float32)
    
    for y in range(0, rows, blockSize):
        for x in range(0, cols, blockSize):
            minMAD = np.finfo(np.float32).max
            bestMV = (0, 0)
            
            for m in range(-searchRange, searchRange + 1):
                for n in range(-searchRange, searchRange + 1):
                    if y+m < 0 or y+m+blockSize > rows or x+n < 0 or x+n+blockSize > cols:
                        continue
                    
                    refBlock = refFrame[y+m:y+m+blockSize, x+n:x+n+blockSize]
                    curBlock = curFrame[y:y+blockSize, x:x+blockSize]
                    MAD = np.abs(refBlock - curBlock).mean()
                    
                    if MAD < minMAD:
                        minMAD = MAD
                        bestMV = (n, m)
                        
            motionVectors[y // blockSize, x // blockSize] = bestMV
            
    return motionVectors
   

def motion_estimation_2dlog_search(refFrame, curFrame, blockSize, searchRange):
    """
     motion estimation by 2D logarithmic search 
     @refFrame: H*W, reference frame with single channel
     @curFrame: H*W, current frame with single channel
     @blockSize: size of block, typically set to 16
     @searchRange: search range [-p, p], typically p is set 15 or 7
    """
    # Initialize variables
    rows, cols = refFrame.shape[:2]
    mv_rows = rows // blockSize
    mv_cols = cols // blockSize
    motionVectors = np.zeros((mv_rows, mv_cols, 2), dtype=np.float32)

    # Loop over every block
    for y in range(0, rows, blockSize):
        for x in range(0, cols, blockSize):
            minMAD = float('inf')
            bestMV = np.array([0, 0])
            center = np.array([x, y], dtype=np.float32) # start point

            for log_scale in range(int(np.log2(searchRange)), -1, -1):
                step_size = 1 << log_scale # step size will be reduced gradually
                
                """ TODO: implement here """
                raise Exception("TODO: implement 2d log. search")

                # loop over search points (8+1 candidate points) 
                # to find a optimal candidate, e.g., [center[0] - step_size, center[1]] is a left point to the center point
                # with a distance step_size, [center[0]+step_size, center[1]] is a right point  to the center.
                # hence, there exist 8 neighours should be compared to the center. 
                        # make sure candidate point within the image
                         
                        # compute MAD between current block and reference block

                        # update minMAD & bestMV

                # update center with optimal candidate

            motionVectors[y//blockSize, x//blockSize] = bestMV
    return motionVectors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Motion estimation using hierarchical search')
    parser.add_argument('-r', '--ref', type=str, required=True,
                        help='Reference frame file name')
    parser.add_argument('-c', '--cur', type=str, required=True,
                        help='Current frame file name')
    parser.add_argument('-b', '--blocksize', type=int, default=16,
                        help='Block size for motion estimation (default: 16)')
    parser.add_argument('-s', '--searchrange', type=int, default=15,
                        help='Search range for motion estimation (default: 15)')
    args = parser.parse_args()

    refFrameFile = args.ref
    curFrameFile = args.cur
    blockSize = args.blocksize
    searchRange = args.searchrange

    # read images
    refFrame = cv2.imread(refFrameFile, 0)
    curFrame = cv2.imread(curFrameFile, 0)
    assert refFrame is not None, "cannot open ref. frame:%s" % refFrameFile
    assert curFrame is not None, "cannot open cur. frame:%s" % curFrameFile
    assert refFrame.shape == curFrame.shape, "ref. frame :{} and cur. frame :{} must have same dimensions !".format(refFrame.shape, curFrame.shape)
    # pad images with zero if needed
    rowsToPad = (blockSize - refFrame.shape[0] % blockSize) % blockSize
    colsToPad = (blockSize - refFrame.shape[1] % blockSize) % blockSize
    if rowsToPad != 0 or colsToPad != 0:
        refFrame = cv2.copyMakeBorder(refFrame, 0, rowsToPad, 0, colsToPad, cv2.BORDER_CONSTANT, value=0)
        curFrame = cv2.copyMakeBorder(curFrame, 0, rowsToPad, 0, colsToPad, cv2.BORDER_CONSTANT, value=0)


    t0 = time.time()
    mv_1 = motion_estimation_full_search(refFrame, curFrame, blockSize, searchRange)
    cost1 = time.time() - t0

    t0 = time.time()
    mv_2 = motion_estimation_2dlog_search(refFrame, curFrame, blockSize, searchRange)
    cost2 = time.time() - t0

    # predict frame

    pred_1 = predict_frame(refFrame, mv_1, blockSize)
    pred_2 = predict_frame(refFrame, mv_2, blockSize)

    mse_1 = compute_mse(pred_1, curFrame)
    mse_2 = compute_mse(pred_2, curFrame)

    print("MSE of full search:", mse_1, " cost time:", cost1, " s.")
    print("MSE of 2D logarithmic search:", mse_2, " cost time:", cost2, " s.")

    mv1_img = visualize_motion_vectors(refFrame, mv_1, blockSize)
    mv2_img = visualize_motion_vectors(refFrame, mv_2, blockSize)

    plt.subplot(231)
    plt.imshow(refFrame, cmap='gray')
    plt.title("Ref. Frame")
    plt.subplot(232)
    plt.imshow(curFrame, cmap='gray')
    plt.title("Cur. Frame")
    plt.subplot(233)
    plt.title("Motion vectors (full search)")
    plt.imshow(mv1_img*10)
    plt.subplot(234)
    plt.imshow(pred_1, cmap='gray')
    plt.title("Pred. Frame (full search)")
    plt.subplot(235)
    plt.imshow(pred_2, cmap='gray')
    plt.title("Pred. Frame (2d log. search)") 
    plt.subplot(236)
    plt.imshow(mv2_img*10)
    plt.title("Motion vectors (2d log. search)")
    plt.show()








