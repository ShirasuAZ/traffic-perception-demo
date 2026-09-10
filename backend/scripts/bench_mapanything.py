import time, sys, torch
from mapanything.models import MapAnything
from mapanything.utils.image import load_images
model = MapAnything.from_pretrained("facebook/map-anything").to("cuda").eval()
img_path = sys.argv[1]
def run(views):
    with torch.no_grad():
        return model.infer(views, memory_efficient_inference=False, use_amp=True, amp_dtype="bf16",
                           apply_mask=True, mask_edges=False, apply_confidence_mask=False)
for res in (518, 392):
    for nviews in (1, 2, 4):
        try:
            views = load_images([img_path]*nviews, resolution_set=res)
        except Exception as e:
            print("res", res, "unsupported:", e); break
        for _ in range(2): run(views)
        torch.cuda.synchronize(); t=time.perf_counter(); N=5
        for _ in range(N): preds = run(views)
        torch.cuda.synchronize(); dt=(time.perf_counter()-t)/N
        p = preds[0]
        K = p["intrinsics"][0].cpu().numpy().round(1).tolist()
        shape = tuple(p["depth_z"].shape)
        mem = torch.cuda.max_memory_allocated()/1e9
        print(f"res={res} views={nviews} depth_shape={shape} time={dt*1000:.0f}ms mem={mem:.1f}GB fx={K[0][0]} cx={K[0][2]}")
    print("keys:", sorted(p.keys()))
