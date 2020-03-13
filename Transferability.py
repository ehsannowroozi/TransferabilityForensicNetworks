########################################
#Title: ON THE TRANSFERABILITY OF ADVERSARIAL EXAMPLES AGAINST CNN-BASED IMAGE FORENSICS

#Authors: Mauro Barni, Kassem Kallas, Ehsan Nowroozi, Benedetta Tondi

#Paper Address: https://ieeexplore.ieee.org/document/8683772

#Cite:
      #@INPROCEEDINGS{8683772,
      #author={M. {Barni} and K. {Kallas} and E. {Nowroozi} and B. {Tondi}},
      #booktitle={ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      #title={On the Transferability of Adversarial Examples against CNN-based Image Forensics},
      #year={2019},
      #volume={},
      #number={},
      #pages={8286-8290},
      #keywords={convolutional neural nets;image forensics;learning (artificial intelligence);object detection;security of data;called adversarial examples;CNN-based image forensic tools;CNN models;security-oriented applications;attack transferability;image forensics applications;forensic analyst;attacker;convolutional neural networks;Adversarial multimedia forensics;adversarial machine learning;adversarial examples;attack transferability;image forensics},
      #doi={10.1109/ICASSP.2019.8683772},
      #ISSN={1520-6149},
      #month={May},}

########################################
import foolbox
from foolbox.models import KerasModel
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
from keras.datasets import mnist
from scipy.misc import imread
from glob import glob
#from utils import force_linear_activation
import cv2
from PIL import Image
import math
import tensorflow as tf
import os

#import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
#keras.backend.set_learning_phase(0)


def show_figures(I, Z, true_score,adv_score):
    plt.figure()

    true_class = np.argmax(true_score)
    adv_class = np.argmax(adv_score)

    plt.subplot(1, 3, 1)
    plt.title('Original (class {}, score {:2.2f})'.format(true_class, true_score[true_class]))
    plt.imshow(I, cmap=plt.get_cmap('gray'))  # division by 255 to convert [0, 255] to [0, 1]
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Adversarial (class {}, score {:2.2f})'.format(adv_class, adv_score[adv_class]))
    plt.imshow(Z, cmap=plt.get_cmap('gray'))  # ::-1 to convert BGR to RGB
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Difference')
    difference = np.double(Z) - np.double(I)  # Z - I
    plt.imshow(difference,cmap=plt.get_cmap('Blues'))# / abs(difference).max() * 0.2 + 0.5)
    plt.axis('off')

    plt.show()
    return


''' Bowen code 7.10'''
def PSNR2MSE(psnr):
    return (1**2)*10**(-psnr/10)
def MSE2PSNR(mse):
    return 10*math.log10(1**2/mse)
'''end Bowen code 7.10'''

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


if __name__ == '__main__':

    def main():

        # Load Keras model
        model = load_model(r'...........................h5')  #First model

        # Switch softmax with linear activations -- per evitare il softmax

        Ptype =  'probabilities' #'logits' # 'probabilities'


        # 64x64, 2 digits
        img_rows, img_cols, img_chans = 128,128, 1
        input_shape = (img_rows, img_cols, img_chans)
        num_classes = 2

        jpeg_quality = 85
        jpeg = 0 # 'true'
        compressJPEG =  0 #'true'

        #---------------------------------------------------------
        #  Load test data and define labels (numImg, 64,64)
        #-----------------------------------------------------------

        images = glob(r'...................\*.png')   #images from first model (Manipulated class)
        label = 0   # label = 1 for Original  and Label = 0 for Manipulated class

        # mismatch model: Load Keras model
        model2 = load_model(r'...................h5')  #load second model
        label2 = 1


        # We compute accuracy based on numebr of images ( 5 ) #Ehsan
        numImg = len(images)

        np.random.seed(1234)
        index = np.random.randint(len(images), size=numImg)

        x_test = np.zeros((numImg, img_rows, img_cols))
        for i in np.arange(numImg):
            img = imread(images[index[i]], flatten=False)  # Flatten=True means convert to gray on the fly
            if compressJPEG:
                img1 = Image.fromarray(img)
                img1.save('temp.jpeg', "JPEG", quality=jpeg_quality)
                img = Image.open('temp.jpeg')
            x_test[i] = img

        # Labels of authentic images = 1 (non-authentic = 0).
        y_test_c = np.tile(label, numImg)


        # Convert labels to one-hot with Keras
        y_test = keras.utils.to_categorical(y_test_c, num_classes)

        # Reshape test data, divide by 255 because net was trained this way
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_chans)

        x_test = x_test.astype('float32')
        x_test /= 255

        # Test legitimate examples
        score = model.evaluate(x_test, y_test, verbose=0)
        #Returns the loss value (of the loss function) & metrics (accuracy ...) values for the model in test mode
        predicted_legitimate_labels = np.argmax(model.predict(x_test), axis=1)

        print('Accuracy on legitimate images (all): {:3.4f}'.format(score[1]))

        y_test_c2 = np.tile(label2, numImg)
        y_test2 = keras.utils.to_categorical(y_test_c2, num_classes)
        #one-hot representation
        score2 = model2.evaluate(x_test, y_test2, verbose=0)
        # Returns the loss value (of the loss function) & metrics (accuracy ...) values for the model in test mode
        #predicted_legitimate_labels2 = np.argmax(model2.predict(x_test), axis=1)
        print('Accuracy on legitimate images (all) by mismatched model: {:3.4f}'.format(score2[1]))

        # ----------------------------------------------------------------------------------------------------------------------
        # Attack the first image of the test set
        # ----------------------------------------------------------------------------------------------------------------------

        # Wrap model
        fmodel = KerasModel(model, bounds=(0, 1), predicts=Ptype)

        # Prepare attack
        #attack = foolbox.attacks.FGSM(fmodel)
        #attack = foolbox.attacks.DeepFoolAttack(fmodel)
        #attack = foolbox.attacks.DeepFoolAttack(fmodel)
        #attack = foolbox.attacks.SaliencyMapAttack(fmodel,threshold=PSNR2MSE(55))
        #attack = foolbox.attacks.LBFGSAttack(fmodel)
        attack = foolbox.attacks.LBFGSAttack(fmodel, threshold=PSNR2MSE(55))  #LBFGS adversarial attack with limit PSNR


        # ------Get data, labels and categorical labels ***only for correctly classified examples***
        l = np.argwhere(predicted_legitimate_labels == y_test_c).shape[0]

        x_test_ok = np.reshape(x_test[np.array(np.argwhere(predicted_legitimate_labels == y_test_c)), :, :, :], (l, img_rows,
                                                                                                                 img_cols,
                                                                                                                img_chans))
        test_ok_index = index[np.array(np.argwhere(predicted_legitimate_labels == y_test_c))]

        # x_test_ok are the images that are correctly classified by the first model since we do not want to attack misclassified images

        y_test_ok = np.reshape(y_test[np.argwhere(predicted_legitimate_labels == y_test_c), :], (l, num_classes))
        y_test_c_ok = np.argmax(y_test_ok, axis=1)


        y_test_c_ok_2 = np.tile(label2, l)
        y_test_ok_2 = keras.utils.to_categorical(y_test_c_ok_2, num_classes)
        score3 = model2.evaluate(x_test_ok, y_test_ok_2, verbose=0)
        predicted_legitimate_labels2 = np.argmax(model2.predict(x_test_ok), axis=1)

        l = np.argwhere(predicted_legitimate_labels2 == y_test_c_ok_2).shape[0]
        x_test_ok = np.reshape(x_test_ok[np.array(np.argwhere(predicted_legitimate_labels2 == y_test_c_ok_2)), :, :, :],
                               (l, img_rows,
                                img_cols,
                                img_chans))
        y_test_ok = np.reshape(y_test_ok[np.argwhere(predicted_legitimate_labels2 == y_test_c_ok_2), :],
                               (l, num_classes))
        y_test_c_ok = np.argmax(y_test_ok, axis=1)

        test_ok_index = np.squeeze(test_ok_index[np.array(np.argwhere(predicted_legitimate_labels2 == y_test_c_ok_2))])

        # ------------------


        # Elaborate n_test adversarial examples ***only for correctly classified examples***
        n_test = l  #Benedetta

        #n_test = l    #Ehsan : You're the man Ehsan

        S = 0
        S_int = 0
        S_jpg  = 0
        avg_Max_dist = 0
        avg_L1_dist = 0
        avg_Max_dist_made_integer = 0
        avg_L1_dist_made_integer = 0
        avg_No_Mod_Pixels = 0
        avg_No_Mod_Pixels_integer_rounding_adv_img = 0
        avg_No_Mod_Pixels_integer_NO_rounding = 0
        PSNR = 0
        t = 0
        avg_psnr = 0
        avg_psnr_int = 0
        psnr_org=0 #for each image
        psnr_Int=0 #for each image
        max_diff_integer=0
        max_diff=0


        adv_images = np.zeros((n_test, img_rows, img_cols, img_chans))
        adv_images_integer = np.zeros((n_test, img_rows, img_cols, img_chans))
        true_labels_cat = []
        for idx in np.arange(n_test):
            #n_test should be less than to the length of x_test_ok
            image = x_test_ok[idx]

            true_labels_cat.append(y_test_ok[idx, :])

            image = image.astype('float32')
            image_original = 255 * image.reshape((img_rows, img_cols))

            if compressJPEG:
                img1 = Image.fromarray(np.uint8(255*image[:,:,0]))
                img1.save('temp.jpeg', "JPEG", quality=jpeg_quality)
                img_reread = Image.open('temp.jpeg')
                image = np.array(img_reread)
                image = np.reshape(image, (img_rows, img_cols, img_chans))


            # Generate adversarial images
            adv_images[idx] = attack(image, y_test_c_ok[idx])

            adversarial_image = 255 * adv_images[idx].reshape((img_rows, img_cols))

            Z = np.uint8(np.round(adversarial_image))

            # Store adversarial integer images
            ##############################################################################

            path1='E:/......................./' #output folder


            cv2.imwrite(os.path.join(path1, os.path.basename(images[test_ok_index[idx]])), Z)

            ##################################################################################
            path2 = '''E:\..................\\'''
            diff_noise=adversarial_image - image_original
            Noise = np.uint8((diff_noise - np.min(diff_noise)) / (np.max(diff_noise) - np.min(diff_noise)))
            cv2.imwrite(path2 + 'adv_Nosie_%d.png' % idx, 255*Noise)
            adv_images_integer[idx] = np.reshape(Z / 255., (img_rows, img_cols, 1))


            # Scores of legitimate and adversarial images for each idx
            scoreTemp = fmodel.predictions(image)
            true_score = foolbox.utils.softmax(scoreTemp)
            true_class = np.argmax(true_score)
            #it is the ground truth true_class according to network 1
            adv_score = foolbox.utils.softmax(fmodel.predictions(adv_images[idx]))
            adv_class = np.argmax(adv_score)
            adv_integer_score = foolbox.utils.softmax(fmodel.predictions(adv_images_integer[idx]))
            adv_integer_class = np.argmax(adv_integer_score)



            print('Image {}. Class changed from {} to {}. The score passes from {} to {}'.format(idx, true_class,
                                                                                                 adv_class, true_score,
                                                                                                 adv_score))

            print('Image Made Integer {}. Class changed from {} to {}. The score passes from {} to {}'.format(idx, true_class,
                                                                                                 adv_integer_class, true_score,
                                                                                                 adv_integer_score))

            # the if below is to solve the strange problem with the prediction of a matrix of nan values...
            if np.any(np.isnan(adv_images[idx])):
                adv_class = true_class
                adv_integer_class = true_class
                t = t + 1
                print('An adversarial image cannot be found!!')


            if true_class == adv_class:
                S = S+1
            if true_class == adv_integer_class:
                S_int = S_int + 1

            # plot image, adv_image and difference
            image_before = 255 * image.reshape((img_rows, img_cols))
            X = np.uint8(image_before) # uint8 non ha effetto di troncamento



            diff = np.double(image_before) - np.double(adversarial_image)

            print('Max distortion adversarial = {:3.4f}; L1 distortion = {:3.4f}'.format(abs(diff).max(),
                                                                                                 abs(diff).sum() / (
                                                                                                             img_rows * img_cols)))
            print('Percentage of modified pixels on integers = {:3.4f}. Percentage of negative modifications  = {:3.4f}'.format(np.count_nonzero(diff)/(img_rows * img_cols), np.count_nonzero(np.double(abs(diff)) - np.double(diff))/(img_rows * img_cols)))


            diff_integer = np.double(X) - np.double(Z)

            max_diff_integer = diff_integer.max()
            max_diff = diff.max()


            path3 = '''E:\Benedetta_for_ICASSP\IMAGE_Diff_Int\\'''
            Noise2 = np.uint8((diff_integer - np.min(diff_integer)) / (np.max(diff_integer) - np.min(diff_integer)))
            cv2.imwrite(path3 + 'adv_Nosie_%d.png' % idx, 255 * Noise2)



            print('Max distortion adversarial integer = {:3.4f}; L1 distortion = {:3.4f}'.format(abs(diff_integer).max(), abs(diff_integer).sum()/(img_rows * img_cols)))

            #show_figures(X,Z,true_score,adv_score)   #Ehsan: Compute PSNR for each Images org and Adversarial integer
            psnr_org=psnr(image_before, adversarial_image)
            print('PSNR = {:3.4f}'.format(abs(psnr_org)))

            psnr_Int = psnr(X, Z)
            print('PSNR (Integer) = {:3.4f}'.format(abs(psnr_Int)))


            # update average distortion
            if true_class != adv_class:
              avg_Max_dist = avg_Max_dist + abs(diff).max()
              avg_L1_dist = avg_L1_dist + abs(diff).sum()/(img_rows * img_cols)
              avg_No_Mod_Pixels = avg_No_Mod_Pixels + np.count_nonzero(diff) / (img_rows * img_cols)
              avg_psnr = avg_psnr + psnr(image_before, adversarial_image)


            if true_class != adv_integer_class:
              avg_Max_dist_made_integer = avg_Max_dist_made_integer + abs(diff_integer).max()
              avg_L1_dist_made_integer = avg_L1_dist_made_integer + abs(diff_integer).sum()/(img_rows * img_cols)
              avg_No_Mod_Pixels_integer_rounding_adv_img = avg_No_Mod_Pixels_integer_rounding_adv_img + np.count_nonzero(diff_integer) / (img_rows * img_cols)  # ????????? why diff ????
              #this after rounding to integer the adversarial image
              avg_No_Mod_Pixels_integer_NO_rounding = avg_No_Mod_Pixels_integer_NO_rounding + np.count_nonzero(diff) / (img_rows * img_cols)
              #this is just without rounding but counting the difference when the true class and the modified class are different
              avg_psnr_int = avg_psnr_int + psnr(X, Z)

            # -------------------------------
            # #Compress JPEG the image and test again
            # -------------------------------

            if jpeg:

                img1 = Image.fromarray(Z)
                img1.save('temp.jpeg', "JPEG", quality= jpeg_quality)
                adv_reread = Image.open('temp.jpeg')
                x_test_comp = np.array(adv_reread)
                x_test_comp = x_test_comp.reshape(img_rows, img_cols, img_chans)
                x_test_comp = x_test_comp.astype('float32')
                x_test_comp /= 255
                adv_reread_score = foolbox.utils.softmax(fmodel.predictions(x_test_comp))
                adv_reread_class = np.argmax(adv_reread_score)
                if true_class == adv_reread_class:
                    S_jpg = S_jpg + 1
                print('Class after JPEG compression {}, with score {}.'.format(adv_reread_class,adv_reread_score))

                x_test_comp = 255* x_test_comp.reshape((img_rows, img_cols))
                print('PSNR = {}'.format(psnr(image_before, x_test_comp)))

                PSNR = psnr(image_before, x_test_comp) + PSNR


        n=n_test-S
        n_int=n_test-S_int
        print('Class for the adversarial unchanged: {} over {}'.format(S,n_test))
        # on how many test images (advesarial) the attack did not work
        print('Class for the adversarial integer unchanged: {} over {}'.format(S_int,n_test))
        # on how many test images (advesarial) integer the attack did not work
        print('Average distortion: max dist {}, L1 dist {}'.format(avg_Max_dist/n,avg_L1_dist/n))
        print('Average distortion (made integer): max dist {}, L1 dist {}'.format(avg_Max_dist_made_integer/n_int,avg_L1_dist_made_integer/n_int))
        print('Average no of modified pixels: {}'.format(avg_No_Mod_Pixels/n))
        print('Average no of modified pixels on integers NO ROUNDING: {}'.format(avg_No_Mod_Pixels_integer_NO_rounding /n_int))
        print('Average no of modified pixels on integers rounding adv_img to int: {}'.format(avg_No_Mod_Pixels_integer_rounding_adv_img / n_int))


        print('The adversarial image cannot be found  {} times over {}'.format(t,n_test))


        if jpeg:
           print('Percentage of adversarial JPEG unchanged with QF {} (the attack is not successful): {}'.format(jpeg_quality, S_jpg/n_test))

        print('Average PSNR distortion for JPEG adversarial images : {}'.format(PSNR/n_test))

        # Evaluate accuracy
        true_labels_cat = np.array(true_labels_cat)
        adv_score = model.evaluate(adv_images, true_labels_cat, verbose=0)
        adv_score_integer= model.evaluate(adv_images_integer, true_labels_cat, verbose=0)

        score_perfect = model.evaluate(x_test_ok, y_test_ok, verbose=0)

        print('Accuracy on legitimate images (all) by N1: {:3.4f}'.format(score[1]))
        print('Accuracy on legitimate images (all) by mismatched model N2: {:3.4f}'.format(score2[1]))  # ????? Score2
        print('Accuracy on legitimate images (only correctly classified, obviously 1) N1: {:3.4f}'.format(score_perfect[1]))
        print('Accuracy on adversarial images N1: {:3.4f}'.format(adv_score[1]))
        print('Attack success rate on adversarial images N1: {:3.4f}'.format(1-adv_score[1]))
        print('Accuracy on adversarial images (made integer) N1: {:3.4f}'.format(adv_score_integer[1]))
        print('Attack success on adversarial images (made integer) N1: {:3.4f}'.format(1-adv_score_integer[1]))
        print('Average PSNR =: {:3.4f}'.format(avg_psnr / n))
        print('Average PSNR (Integer) =: {:3.4f}'.format(avg_psnr_int / n_int))

        # SECOND PART
        # Load the second model and test the adversarial images
        # Label
        #label3 = np.abs(1 - label2)  # it may be different from label because of the differences in the model.

        # Labels
        y_test_c = np.tile(label2, n_test)

        # Convert labels to one-hot with Keras
        y_test2 = keras.utils.to_categorical(y_test_c, num_classes)

        # Test
        adv_score_mismatch = model2.evaluate(adv_images, y_test2, verbose=0)

        # here Ehsan we need to evaluate model 2 in the same way but not on adv_images ... on adv_images_integer

        adv_score_mismatch_on_integer = model2.evaluate(adv_images_integer, y_test2, verbose=0)



        print('Accuracy on adversarial images with the mismatched model N2: {:3.4f}'.format(adv_score_mismatch[1]))
        print('Attack success rate on adversarial images with the mismatched model N2: {:3.4f}'.format(1-adv_score_mismatch[1]))

        print('Accuracy on adversarial images with the mismatched model (Integer) N2: {:3.4f}'.format(adv_score_mismatch_on_integer[1]))
        print('Attack success rate on adversarial images with the mismatched model (Integer) N2: {:3.4f}'.format(1-adv_score_mismatch_on_integer[1]))


    # Force code to run on CPU so that it does not bother concurrent tasks that use GPU(s)
    #with tf.device('/cpu:0'):  # ('/cpu:0', '/gpu:0', '/gpu:2'): # ('/cpu:0'):
    main()
