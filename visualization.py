from options import Options
from model import Model, mse_to_psnr
import matplotlib.pyplot as plt


def eval_baseline():
    opt = Options().parse()
    # opt.use_haptic = True
    # opt.use_behavior = True
    # opt.use_audio = True
    opt.baseline = True
    opt.sequence_length = 20
    print("Model Config: ", opt)
    model = Model(opt)
    # model.load_weight("/home/golf/code/multi_modal_video_prediction/weight_use_haptic_1/net_epoch_29.pth")
    model.load_weight("/home/golf/code/multi_modal_video_prediction/weight_baseline_1/net_epoch_29.pth")
    # print(model.evaluate(0))
    return model.evaluate(0, keep_frame=True)

def eval_proposed():
    opt = Options().parse()
    opt.use_haptic = True
    opt.use_behavior = True
    opt.use_audio = True
    opt.sequence_length = 20
    # opt.baseline = True
    print("Model Config: ", opt)
    model = Model(opt)
    # model.load_weight("/home/golf/code/multi_modal_video_prediction/weight_use_haptic_1/net_epoch_29.pth")
    model.load_weight("/home/golf/code/multi_modal_video_prediction/weight_use_haptic_1/net_epoch_29.pth")
    # print(model.evaluate(0))
    return model.evaluate(0, keep_frame=True)

if __name__ == '__main__':
    psnr_baseline = [mse_to_psnr(mse) for mse in eval_baseline()]
    psnr_propsed = [mse_to_psnr(mse) for mse in eval_proposed()]
    frames = range(5, 21)
    plt.plot(frames, psnr_baseline, 'r')
    plt.plot(frames, psnr_propsed, 'b')
    plt.show()
