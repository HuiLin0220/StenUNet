import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft

plt.rcParams['image.cmap'] = "gray"

# Homomorphic filter class
class HomomorphicFilter:
    """
    High-frequency filters implemented:
        butterworth
        gaussian
    Attributes:
        gH, gL: Floats used on emphasis filter:
            H = gL + (gH-gL)*H

Attenuate the contribution made by the low frequencies(illumination) and amplify the contribution made by high frequencies(reflectance).
The net result is simultaneaous dynamic range compression and contrast enhacement.
The costant C control the sharpness of the function as it transition between deltaL and deltaH.
If gH>=1 and 0<gL<1 the high frequencies are amplified and the low frequencies are cutted off.
gL is also used to preserve the tonality of the image.
    """

    def __init__(self, gH=1.5, gL=0.5):
        self.gH = float(gH)
        self.gL = float(gL)

    # D(u,v)
    def __Duv(self, I_shape):
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U - P) ** 2 + (V - Q) ** 2) ** (1 / 2)).astype(np.dtype('d'))

        return Duv

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):  # crea un highpass filter
        Duv = self.__Duv(I_shape)
        n = filter_params[2]
        c = filter_params[1]
        D0 = filter_params[0]
        h = 1 / (1 + ((c * Duv) / D0) ** (2 * n))  # lowpass filter
        H = (1 - h)
        return H

    def __gaussian_filter(self, I_shape, filter_params):  # crea un highpass filter
        Duv = self.__Duv(I_shape)
        c = filter_params[1]
        D0 = filter_params[0]
        h = np.exp((-c * (Duv ** 2) / (2 * (D0 ** 2))))  # lowpass filter
        H = (1 - h)

        return H

    def __plot_Filter(self, I, H, filter_params):
        I_shape = I.shape
        params = ', gH: ' + str(self.gH) + ', gL: ' + str(self.gL) + '\n D0: ' + str(filter_params[0]) + ', c: ' + str(
            filter_params[1]) + ', order: ' + str(filter_params[2])
        # plt.title('Transfer function' + params)
        # plt.ylabel('H(x,y)')
        # plt.xlabel('D(u,v)')
        if I_shape[0] > I_shape[1]:
            pass
            plt.plot(self.__Duv(I_shape)[int(I_shape[1] / 2)], H[int(I_shape[1] / 2)]);
            # plt.show()
            # plt.plot(H[int(I_shape[1] / 2)]);

        else:
            pass
            # plt.plot(self.__Duv(I_shape)[int(I_shape[0] / 2)], H[int(I_shape[0] / 2)]);
            # plt.show()
            # plt.plot(H[int(I_shape[0] / 2)]);
        # plt.title('HighPass Filter')
        # plt.ylabel('H(x,y)')
        # plt.xlabel('Coordinate of the matrix')
        # plt.show()

    # Methods
    def __apply_filter(self, I, H, params):

        if self.gH < 1 or self.gL >= 1:
            H = H
        else:
            H = ((self.gH - self.gL) * H + self.gL)

        I_filtered = H * I
        self.__plot_Filter(I, H, params)

        return I_filtered

    def apply_filter(self, I, filter_params=(12, 1, 2), filter_='butterworth', H=None):
        """
        Method to apply homormophic filter on an image
        Attributes:
            I: Single channel image
            filter_params: Parameters to be used on filters:
                butterworth:
                    filter_params[0]: Cutoff frequency
                    filter_params[1]: c
                    filter_params[2]: Order of filter

                gaussian:
                    filter_params[0]: Cutoff frequency
                    filter_params[1]: c
            filter: Choose of the filter, options:
                butterworth
                gaussian
                external
            H: Used to pass external filter
        """

        #  Validating image
        if len(I.shape) != 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain
        I_log = np.log1p(np.array(I, dtype='d'))
        I_fft = np.fft.fft2(I_log)
        I_fft = np.fft.fftshift(I_fft)
        # Filters
        if filter_ == 'butterworth':
            H = self.__butterworth_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter_ == 'gaussian':
            H = self.__gaussian_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter_ == 'external':
            print('external')
            if len(H.shape) != 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')

        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I=I_fft, H=H, params=filter_params)
        I_fft_filt = np.fft.fftshift(I_fft_filt)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.expm1(np.real(I_filt))

        Imax = (np.max(I))
        Imin = (np.min(I))
        I = 255 * ((I - Imin) / (Imax - Imin))  # Image is normalized

        return I
# End of class HomomorphicFilter

def homomorphic_filter(img, D0=12, n=2):
    log_image = np.log1p(img.copy())  # Apply logarithm to the image

    frequency_image = fft.fftshift(fft.fft2(log_image))

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(-crow, crow)
    v = np.arange(-ccol, ccol)
    uu, vv = np.meshgrid(u, v)
    D = np.sqrt(uu ** 2 + vv ** 2)
    H = 1 / (1 + (D / D0) ** (2 * n))

    filtered_frequency_image = frequency_image * H

    filtered_image = np.real(fft.ifft2(fft.ifftshift(filtered_frequency_image)))

    illumination_pattern = np.exp(filtered_image)

    return filtered_image, illumination_pattern


def adjust_scale(image):
    # Find the minimum and maximum values in the array
    min_val = np.min(image)
    max_val = np.max(image)

    # Scale the array to the range 0 to 255
    adjusted_image = ((image - min_val) / (max_val - min_val)) * 255

    # Convert the values to integers
    adjusted_image

    return adjusted_image


def normalize(img, M0=128, VAR0=100):  # Desired variance):
    uniform_image = img.copy()

    mean = np.mean(uniform_image)
    variance = np.var(uniform_image)

    mask = uniform_image > mean

    normalized_image = np.zeros_like(uniform_image, dtype=np.float32)
    normalized_image[mask] = M0 + np.sqrt(VAR0 * (uniform_image[mask] - mean) ** 2 / variance)
    normalized_image[~mask] = M0 - np.sqrt(VAR0 * (uniform_image[~mask] - mean) ** 2 / variance)
    normalized_image = adjust_scale(normalized_image) #np.clip(normalized_image, 0, 255).astype(np.uint8)

    return normalized_image


def homomorphic_filter(img, D0=12, n=2):
    a, b = 0.5, 1.5
    log_image = np.log1p(img.copy())  # Apply logarithm to the image

    frequency_image = fft.fftshift(fft.fft2(log_image))

    rows, cols = img.shape
    P, Q = rows // 2, cols // 2
    U, V = np.meshgrid(range(rows), range(cols), sparse=False,   indexing='ij')
    Duv = (((U-P)**2+(V-Q)**2)).astype(float)
    H = 1/(1+((c*Duv)/D0)**(2*n))
    H = 1 - H

    H = np.fft.fftshift(H)


    if b<1 or a>=1:
        H = H
    else:
        H = ((b-a)*H+a)

    I_filtered=H * frequency_image

    I_fft_filt=np.fft.fftshift(I_filtered)
    I_filt = np.fft.ifft2(I_fft_filt)
    I=np.expm1(np.real(I_filt))

    return I



