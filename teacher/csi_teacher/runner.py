"""rtmlib 어댑터 — RTMDet-m(사람)→RTMPose-m (설계 §7). rtmlib 의존은 이 모듈 격리.

PoseRunner 프로토콜:
  detect(frame_bgr) -> (N,5) f32 [x1,y1,x2,y2,score] — floor 이상 전부, score 내림차순
  pose(frame_bgr, bbox4) -> (17,3) f32 [x_px, y_px, score]

det_thr 필터는 labels.label_frame 몫 — floor(0.05)만 깔아 임계 변경에 재추론 불필요.

실측(rtmlib 0.0.15):
  - RTMDet ONNX(coco-obj365-person) 출력: dets (1,N,5)[x1,y1,x2,y2,score] +
    labels (1,N)[cls_int] — NMS 포함 export. ratio는 preprocess 반환, 역변환 필요.
  - RTMDet.__call__은 점수 버리고 박스만 반환 → detect()에서
    preprocess+inference 직접 호출.
  - session 속성명: BaseTool.session (onnxruntime.InferenceSession).
"""
import numpy as np

# mmpose deployee ONNX — 계획 URL(projects/rtmo/...rtmdet_m_640-8xb32_coco-person)은 404,
# 실존 경로는 rtmposev1/onnx_sdk의 coco-obj365-person export (404 시 rtmlib README 최신으로 교체)
DET_ONNX = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.zip")
POSE_ONNX = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
             "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip")
DET_FLOOR = 0.05
PERSON_CLS = 0


class RtmlibRunner:
    det_model = "RTMDet-m"
    pose_model = "RTMPose-m"

    def __init__(self, device="cuda"):
        try:
            # pip nvidia-*-cu12 라이브러리를 ORT가 찾도록 선로드 (ORT 1.21+ API).
            # 미호출 시 libcublasLt.so.12 미발견 → CUDA EP 로드 실패, CPU 폴백 (WSL2 실측).
            import onnxruntime as ort
            ort.preload_dlls()
        except Exception:
            pass                                 # 구버전 ORT — 시스템 CUDA에 맡김
        from rtmlib import RTMDet, RTMPose
        self._det = RTMDet(onnx_model=DET_ONNX, model_input_size=(640, 640),
                           backend="onnxruntime", device=device)
        self._pose = RTMPose(onnx_model=POSE_ONNX, model_input_size=(192, 256),
                             to_openpose=False, backend="onnxruntime", device=device)
        if device == "cuda":
            for s in (self._det.session, self._pose.session):
                prov = s.get_providers()
                if "CUDAExecutionProvider" not in prov:
                    # ORT는 EP 로드 실패 시 CPU로 폴백해 세션을 만든다 — auto 폴백 로직이 잡도록 격상
                    raise RuntimeError(f"CUDA EP 비활성 — providers={prov}")

    def detect(self, frame_bgr):
        """RTMDet 추론 → (N,5) [x1,y1,x2,y2,score], floor 이상 전부, score 내림차순.

        RTMDet ONNX(coco-obj365-person)는 NMS 내장 export:
          outputs[0] = dets  (1, N, 5) [x1,y1,x2,y2,score] — 모델 입력 좌표계
          outputs[1] = labels (1, N)   int64 class id
        preprocess가 반환하는 ratio로 역변환해 원본 픽셀 좌표계로 복원.
        """
        img, ratio = self._det.preprocess(frame_bgr)
        outputs = self._det.inference(img)
        dets, labels = outputs[0], outputs[1]   # (1,N,5), (1,N)
        keep = (labels[0].astype(int) == PERSON_CLS) & (dets[0, :, 4] >= DET_FLOOR)
        out = dets[0][keep].astype(np.float32)
        out[:, :4] /= ratio
        return out[np.argsort(-out[:, 4])]

    def pose(self, frame_bgr, bbox):
        """RTMPose 추론 → (17,3) [x_px, y_px, score]."""
        kpts, scores = self._pose(frame_bgr, bboxes=[np.asarray(bbox, np.float32)])
        return np.concatenate([kpts[0], scores[0][:, None]], axis=1).astype(np.float32)


def make_runner(device="auto"):
    if device != "auto":
        return RtmlibRunner(device=device)
    try:
        r = RtmlibRunner(device="cuda")
        print("runner: CUDA EP")
        return r
    except Exception as e:
        print(f"경고: CUDA 실패({type(e).__name__}: {e}) — CPU 폴백 (느림)")
        return RtmlibRunner(device="cpu")
