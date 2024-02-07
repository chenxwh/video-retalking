# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import sys
import shutil
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat
import torch
import cv2
from cog import BasePredictor, Input, Path

sys.path.insert(0, "third_part")
sys.path.insert(0, "third_part/GPEN")
sys.path.insert(0, "third_part/GFPGAN")

# 3dmm extraction
from third_part.face3d.util.preprocess import align_img
from third_part.face3d.util.load_mats import load_lm3d
from third_part.face3d.extract_kp_videos import KeypointExtractor

# face enhancement
from third_part.GPEN.gpen_face_enhancer import FaceEnhancement
from third_part.GFPGAN.gfpgan import GFPGANer

# expression control
from third_part.ganimation_replicate.model.ganimation import GANimationModel

from utils import audio
from utils.ffhq_preprocess import Croper
from utils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from utils.inference_utils import (
    Laplacian_Pyramid_Blending_with_mask,
    face_detect,
    load_model,
    options,
    split_coeff,
    trans_image,
    transform_semantic,
    find_crop_norm_ratio,
    load_face3d_net,
    exp_aus_dict,
)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # The weights are downloaded to ./checkpoints first, with
        # wget https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/<file_name>
        self.enhancer = FaceEnhancement(
            base_dir="checkpoints",
            size=512,
            model="GPEN-BFR-512",
            use_sr=False,
            sr_model="rrdb_realesrnet_psnr",
            channel_multiplier=2,
            narrow=1,
            device="cuda",
        )
        self.restorer = GFPGANer(
            model_path="checkpoints/GFPGANv1.3.pth",
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            bg_upsampler=None, bs_upscale_67
        )
        self.croper = Croper("checkpoints/shape_predictor_68_face_landmarks.dat")
        self.kp_extractor = KeypointExtractor()
        self.net_recon = load_face3d_net(
            "checkpoints/face3d_pretrain_epoch_20.pth", "cuda"
        )
        self.lm3d_std = load_lm3d("checkpoints/BFM")

    def predict(
        self,
        face: Path = Input(description="Input video file of a talking-head."),
        input_audio: Path = Input(
            description="Input audio file. Avoid special symbol in the filename as it may cause ffmpeg erros."
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        device = "cuda"
        args = argparse.Namespace(
            DNet_path="checkpoints/DNet.pt",
            LNet_path="checkpoints/LNet.pth",
            ENet_path="checkpoints/ENet.pth",
            face3d_net_path="checkpoints/face3d_pretrain_epoch_20.pth",
            face=str(face),
            audio=str(input_audio),
            exp_img="neutral",
            outfile=None,
            fps=25,
            pads=[0, 20, 0, 0],
            face_det_batch_size=4,
            LNet_batch_size=16,
            img_size=384,
            crop=[0, -1, 0, -1],
            box=[-1, -1, -1, -1],
            nosmooth=False,
            static=False,
            up_face="original",
            one_shot=False,
            without_rl1=False,
            tmp_dir="cog_temp",
            re_preprocess=False,
        )

        if os.path.exists(args.tmp_dir):
            shutil.rmtree(args.tmp_dir)
        os.makedirs(args.tmp_dir)

        output_file = "/tmp/output.mp4"
        if os.path.exists(output_file):
            os.remove(output_file)

        if args.face.split(".")[1] in ["jpg", "png", "jpeg"]:
            full_frames = [cv2.imread(args.face)]
            args.static = True
            fps = args.fps
        else:
            video_stream = cv2.VideoCapture(args.face)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            full_frames = []
            while True:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                y1, y2, x1, x2 = args.crop
                if x2 == -1:
                    x2 = frame.shape[1]
                if y2 == -1:
                    y2 = frame.shape[0]
                frame = frame[y1:y2, x1:x2]
                full_frames.append(frame)
            video_stream.release()

        full_frames_RGB = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in full_frames
        ]
        full_frames_RGB, crop, quad = self.croper.crop(full_frames_RGB, xsize=512)

        clx, cly, crx, cry = crop
        lx, ly, rx, ry = quad
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        oy1, oy2, ox1, ox2 = (
            cly + ly,
            min(cly + ry, full_frames[0].shape[0]),
            clx + lx,
            min(clx + rx, full_frames[0].shape[1]),
        )
        # original_size = (ox2 - ox1, oy2 - oy1)
        frames_pil = [
            Image.fromarray(cv2.resize(frame, (256, 256))) for frame in full_frames_RGB
        ]

        # get the landmark according to the detected face.
        print("[Step 1] Landmarks Extraction in Video.")
        lm = self.kp_extractor.extract_keypoint(
            frames_pil, f"{args.tmp_dir}/landmarks.txt"
        )

        video_coeffs = []
        for idx in tqdm(
            range(len(frames_pil)), desc="[Step 2] 3DMM Extraction In Video:"
        ):
            frame = frames_pil[idx]
            W, H = frame.size
            lm_idx = lm[idx].reshape([-1, 2])
            if np.mean(lm_idx) == -1:
                lm_idx = (self.lm3d_std[:, :2] + 1) / 2.0
                lm_idx = np.concatenate([lm_idx[:, :1] * W, lm_idx[:, 1:2] * H], 1)
            else:
                lm_idx[:, -1] = H - 1 - lm_idx[:, -1]

            trans_params, im_idx, lm_idx, _ = align_img(frame, lm_idx, self.lm3d_std)
            trans_params = np.array(
                [float(item) for item in np.hsplit(trans_params, 5)]
            ).astype(np.float32)
            im_idx_tensor = (
                torch.tensor(np.array(im_idx) / 255.0, dtype=torch.float32)
                .permute(2, 0, 1)
                .to(device)
                .unsqueeze(0)
            )
            with torch.no_grad():
                coeffs = split_coeff(self.net_recon(im_idx_tensor))

            pred_coeff = {key: coeffs[key].cpu().numpy() for key in coeffs}
            pred_coeff = np.concatenate(
                [
                    pred_coeff["id"],
                    pred_coeff["exp"],
                    pred_coeff["tex"],
                    pred_coeff["angle"],
                    pred_coeff["gamma"],
                    pred_coeff["trans"],
                    trans_params[None],
                ],
                1,
            )
            video_coeffs.append(pred_coeff)
        semantic_npy = np.array(video_coeffs)[:, 0]

        if args.exp_img == "smile":
            expression = torch.tensor(
                loadmat("checkpoints/expression.mat")["expression_mouth"]
            )[0]
        else:
            print("using expression center")
            expression = torch.tensor(
                loadmat("checkpoints/expression.mat")["expression_center"]
            )[0]

        # load DNet, model(LNet and ENet)
        D_Net, model = load_model(args, device)

        imgs = []
        for idx in tqdm(
            range(len(frames_pil)),
            desc="[Step 3] Stabilize the expression In Video:",
        ):
            if args.one_shot:
                source_img = trans_image(frames_pil[0]).unsqueeze(0).to(device)
                semantic_source_numpy = semantic_npy[0:1]
            else:
                source_img = trans_image(frames_pil[idx]).unsqueeze(0).to(device)
                semantic_source_numpy = semantic_npy[idx : idx + 1]
            ratio = find_crop_norm_ratio(semantic_source_numpy, semantic_npy)
            coeff = transform_semantic(semantic_npy, idx, ratio).unsqueeze(0).to(device)

            # hacking the new expression
            coeff[:, :64, :] = expression[None, :64, None].to(device)
            with torch.no_grad():
                output = D_Net(source_img, coeff)
            img_stablized = np.uint8(
                (
                    output["fake_image"]
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .cpu()
                    .clamp_(-1, 1)
                    .numpy()
                    + 1
                )
                / 2.0
                * 255
            )
            imgs.append(cv2.cvtColor(img_stablized, cv2.COLOR_RGB2BGR))

        del D_Net
        torch.cuda.empty_cache()

        if not args.audio.endswith(".wav"):
            command = "ffmpeg -loglevel error -y -i {} -strict -2 {}".format(
                args.audio, f"{args.tmp_dir}/input_audio.wav"
            )
            subprocess.call(command, shell=True)
            args.audio = f"{args.tmp_dir}/input_audio.wav"
        wav = audio.load_wav(args.audio, 16000)
        mel = audio.melspectrogram(wav)
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                "Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again"
            )

        mel_step_size, mel_idx_multiplier, i, mel_chunks = 16, 80.0 / fps, 0, []
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size :])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1

        print("[Step 4] Load audio; Length of mel chunks: {}".format(len(mel_chunks)))
        imgs = imgs[: len(mel_chunks)]
        full_frames = full_frames[: len(mel_chunks)]
        lm = lm[: len(mel_chunks)]

        imgs_enhanced = []
        for idx in tqdm(range(len(imgs)), desc="[Step 5] Reference Enhancement"):
            img = imgs[idx]
            pred, _, _ = self.enhancer.process(
                img, img, face_enhance=True, possion_blending=False
            )
            imgs_enhanced.append(pred)
        gen = datagen(
            args.tmp_dir,
            imgs_enhanced.copy(),
            mel_chunks,
            full_frames,
            args,
            (oy1, oy2, ox1, ox2),
        )

        frame_h, frame_w = full_frames[0].shape[:-1]
        out = cv2.VideoWriter(
            f"{args.tmp_dir}/result.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_w, frame_h),
        )

        if args.up_face != "original":
            instance = GANimationModel()
            instance.initialize()
            instance.setup()

        # kp_extractor = KeypointExtractor()
        for i, (
            img_batch,
            mel_batch,
            frames,
            coords,
            img_original,
            f_frames,
        ) in enumerate(
            tqdm(
                gen,
                desc="[Step 6] Lip Synthesis:",
                total=int(np.ceil(float(len(mel_chunks)) / args.LNet_batch_size)),
            )
        ):
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(
                device
            )
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(
                device
            )
            img_original = (
                torch.FloatTensor(np.transpose(img_original, (0, 3, 1, 2))).to(device)
                / 255.0
            )  # BGR -> RGB

            with torch.no_grad():
                incomplete, reference = torch.split(img_batch, 3, dim=1)
                pred, low_res = model(mel_batch, img_batch, reference)
                pred = torch.clamp(pred, 0, 1)

                if args.up_face in ["sad", "angry", "surprise"]:
                    tar_aus = exp_aus_dict[args.up_face]
                else:
                    pass

                if args.up_face == "original":
                    cur_gen_faces = img_original
                else:
                    test_batch = {
                        "src_img": torch.nn.functional.interpolate(
                            (img_original * 2 - 1), size=(128, 128), mode="bilinear"
                        ),
                        "tar_aus": tar_aus.repeat(len(incomplete), 1),
                    }
                    instance.feed_batch(test_batch)
                    instance.forward()
                    cur_gen_faces = torch.nn.functional.interpolate(
                        instance.fake_img / 2.0 + 0.5, size=(384, 384), mode="bilinear"
                    )

                if args.without_rl1 is not False:
                    incomplete, reference = torch.split(img_batch, 3, dim=1)
                    mask = torch.where(
                        incomplete == 0,
                        torch.ones_like(incomplete),
                        torch.zeros_like(incomplete),
                    )
                    pred = pred * mask + cur_gen_faces * (1 - mask)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

            torch.cuda.empty_cache()
            for p, f, xf, c in zip(pred, frames, f_frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                ff = xf.copy()
                ff[y1:y2, x1:x2] = p

                # month region enhancement by GFPGAN
                cropped_faces, restored_faces, restored_img = self.restorer.enhance(
                    ff, has_aligned=False, only_center_face=True, paste_back=True
                )
                # 0,   1,   2,   3,   4,   5,   6,   7,   8,  9, 10,  11,  12,
                mm = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0]
                mouse_mask = np.zeros_like(restored_img)
                tmp_mask = self.enhancer.faceparser.process(
                    restored_img[y1:y2, x1:x2], mm
                )[0]
                mouse_mask[y1:y2, x1:x2] = (
                    cv2.resize(tmp_mask, (x2 - x1, y2 - y1))[:, :, np.newaxis] / 255.0
                )

                height, width = ff.shape[:2]
                restored_img, ff, full_mask = [
                    cv2.resize(x, (512, 512))
                    for x in (restored_img, ff, np.float32(mouse_mask))
                ]
                img = Laplacian_Pyramid_Blending_with_mask(
                    restored_img, ff, full_mask[:, :, 0], 10
                )
                pp = np.uint8(cv2.resize(np.clip(img, 0, 255), (width, height)))

                pp, orig_faces, enhanced_faces = self.enhancer.process(
                    pp, xf, bbox=c, face_enhance=False, possion_blending=True
                )
                out.write(pp)
        out.release()

        command = "ffmpeg -loglevel error -y -i {} -i {} -strict -2 -q:v 1 {}".format(
            args.audio, f"{args.tmp_dir}/result.mp4", output_file
        )

        subprocess.call(command, shell=True)
        return Path(output_file)


# frames:256x256, full_frames: original size
def datagen(tmp_dir, frames, mels, full_frames, args, cox):
    img_batch, mel_batch, frame_batch, coords_batch, ref_batch, full_frame_batch = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    base_name = args.face.split("/")[-1]
    refs = []
    image_size = 256

    # original frames
    kp_extractor = KeypointExtractor()
    fr_pil = [Image.fromarray(frame) for frame in frames]
    lms = kp_extractor.extract_keypoint(
        fr_pil, f"{tmp_dir}/" + base_name + "x12_landmarks.txt"
    )
    frames_pil = [
        (lm, frame) for frame, lm in zip(fr_pil, lms)
    ]  # frames is the cropped version of modified face
    crops, orig_images, quads = crop_faces(
        image_size, frames_pil, scale=1.0, use_fa=True
    )
    inverse_transforms = [
        calc_alignment_coefficients(
            quad + 0.5,
            [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]],
        )
        for quad in quads
    ]
    del kp_extractor.detector

    oy1, oy2, ox1, ox2 = cox
    face_det_results = face_detect(full_frames, args, jaw_correction=True)

    for inverse_transform, crop, full_frame, face_det in zip(
        inverse_transforms, crops, full_frames, face_det_results
    ):
        imc_pil = paste_image(
            inverse_transform,
            crop,
            Image.fromarray(
                cv2.resize(
                    full_frame[int(oy1) : int(oy2), int(ox1) : int(ox2)], (256, 256)
                )
            ),
        )

        ff = full_frame.copy()
        ff[int(oy1) : int(oy2), int(ox1) : int(ox2)] = cv2.resize(
            np.array(imc_pil.convert("RGB")), (ox2 - ox1, oy2 - oy1)
        )
        oface, coords = face_det
        y1, y2, x1, x2 = coords
        refs.append(ff[y1:y2, x1:x2])

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face = refs[idx]
        oface, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))
        oface = cv2.resize(oface, (args.img_size, args.img_size))

        img_batch.append(oface)
        ref_batch.append(face)
        mel_batch.append(m)
        coords_batch.append(coords)
        frame_batch.append(frame_to_save)
        full_frame_batch.append(full_frames[idx].copy())

        if len(img_batch) >= args.LNet_batch_size:
            img_batch, mel_batch, ref_batch = (
                np.asarray(img_batch),
                np.asarray(mel_batch),
                np.asarray(ref_batch),
            )
            img_masked = img_batch.copy()
            img_original = img_batch.copy()
            img_masked[:, args.img_size // 2 :] = 0
            img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.0
            mel_batch = np.reshape(
                mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
            )

            yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch
            (
                img_batch,
                mel_batch,
                frame_batch,
                coords_batch,
                img_original,
                full_frame_batch,
                ref_batch,
            ) = ([], [], [], [], [], [], [])

    if len(img_batch) > 0:
        img_batch, mel_batch, ref_batch = (
            np.asarray(img_batch),
            np.asarray(mel_batch),
            np.asarray(ref_batch),
        )
        img_masked = img_batch.copy()
        img_original = img_batch.copy()
        img_masked[:, args.img_size // 2 :] = 0
        img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.0
        mel_batch = np.reshape(
            mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
        )
        yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch
