import generator
import pattern



if __name__ == '__main__':

    checker1 = pattern.Checker(100, 10)
    checker1.show()

    Circle1 = pattern.Circle(1024, 200, (512, 256))
    Circle1.show()

    Spectrum1 = pattern.Spectrum(255)
    Spectrum1.show()

    data1 = generator.ImageGenerator('./exercise_data/', './Labels.json', batch_size=12, image_size=[32, 32, 3],
                                     rotation=False, mirroring=False, shuffle=False)
    show = data1.show

