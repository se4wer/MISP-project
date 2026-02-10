"""
Generate the PDF report for the Generalized Tikhonov Regularization assignment.
Produces: D:\\Math project\\report.pdf

Requires: fpdf2, numpy, scipy, matplotlib
"""
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.special import j1
from fpdf import FPDF
import os

OUT_DIR = r'D:\Math project'
FIG_DIR = os.path.join(OUT_DIR, '_report_figs')
os.makedirs(FIG_DIR, exist_ok=True)

np.random.seed(42)

# ========================================================
# Re-run the computation (same as notebook)
# ========================================================

def shepp_logan_phantom(N=256):
    img = np.zeros((N, N), dtype=np.float64)
    x = np.linspace(-1, 1, N); y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    ellipses = [
        (1.0,0.69,0.92,0,0,0),(-0.8,0.6624,0.874,0,-0.0184,0),
        (-0.2,0.11,0.31,-0.22,0,-18),(-0.2,0.16,0.41,0.22,0,18),
        (0.1,0.21,0.25,0,0.35,0),(0.1,0.046,0.046,0,0.1,0),
        (0.1,0.046,0.046,0,-0.1,0),(0.1,0.046,0.023,-0.08,-0.605,0),
        (0.1,0.023,0.023,0,-0.606,0),(0.1,0.023,0.046,0.06,-0.605,0),
    ]
    for rho,a,b,x0,y0,td in ellipses:
        t=np.radians(td); ct,st=np.cos(t),np.sin(t)
        Xr=ct*(X-x0)+st*(Y-y0); Yr=-st*(X-x0)+ct*(Y-y0)
        img[(Xr/a)**2+(Yr/b)**2<=1]+=rho
    return img

def compute_psnr(orig, recon):
    mse=np.mean((orig-recon)**2)
    if mse<1e-15: return float('inf')
    return 10*np.log10(np.max(orig)**2/mse)

def forward_problem(f, K, snr):
    fh=fft2(f); gc=np.real(ifft2(fh*K))
    sn=10**(-snr/20)*np.std(gc); w=np.random.normal(0,sn,f.shape)
    g=gc+w; return g,fft2(g),gc,sn,w

def tikhonov(gh,K,Psq,mu):
    return np.real(ifft2(np.conj(K)*gh/(np.abs(K)**2+mu*Psq)))

def spectral_window(gh,K,Om,omf):
    W=(omf<Om).astype(float); fh=np.zeros_like(gh); m=W>0
    fh[m]=W[m]*gh[m]/K[m]; return np.real(ifft2(fh))

def bias_variance(K,Psq,f,sigma,mus):
    Np=K.shape[0]*K.shape[1]; fh=fft2(f); fs=np.abs(fh)**2; Ks=np.abs(K)**2
    b=np.zeros(len(mus)); v=np.zeros(len(mus))
    for i,mu in enumerate(mus):
        d=Ks+mu*Psq; R=np.conj(K)/d
        b[i]=(1/Np)*np.sum(np.abs(R*K-1)**2*fs)
        v[i]=sigma**2*np.sum(np.abs(R)**2)
    return b,v,b+v

N=256; f_true=shepp_logan_phantom(N)
M=N; xr=np.arange(-N//2,N//2); yr=np.arange(-M//2,M//2)
oX,oY=np.meshgrid(xr,yr); om=np.sqrt(oX**2+oY**2)

# Blur kernels
sig_b=3.0; Kg_c=np.exp(-om**2/(2*sig_b**2)); Kg=ifftshift(Kg_c)
L=15; ma=oX*L/2; Km_c=np.exp(-1j*(L/2)*oX)*np.sinc(ma/np.pi); Km=ifftshift(Km_c)
R_d=8.0; da=om*R_d; Ko_c=np.where(da>1e-10,2*j1(da)/da,1.0); Ko=ifftshift(Ko_c)

# Observations
g40,gh40,gcl,s40,_=forward_problem(f_true,Kg,40)
g20,gh20,_,s20,_=forward_problem(f_true,Kg,20)
gm40,ghm40,_,sm40,_=forward_problem(f_true,Km,40)
go40,gho40,_,so40,_=forward_problem(f_true,Ko,40)

# Penalties
P_L2=ifftshift(np.ones((N,M)))
P_H1=ifftshift(oX**2+oY**2)
P_H2=ifftshift((oX**2+oY**2)**2)
pens={'L2':P_L2,'H1':P_H1,'H2':P_H2}

mu_h=1e-3; mu_l=1e-1
rh={n:tikhonov(gh40,Kg,p,mu_h) for n,p in pens.items()}
rl={n:tikhonov(gh20,Kg,p,mu_l) for n,p in pens.items()}
rm={n:tikhonov(ghm40,Km,p,mu_h) for n,p in pens.items()}
ro={n:tikhonov(gho40,Ko,p,mu_h) for n,p in pens.items()}

omf=ifftshift(om)
cuts=[20,40,60,80]
wr={O:spectral_window(gh40,Kg,O,omf) for O in cuts}
best_O=max(cuts,key=lambda O:compute_psnr(f_true,wr[O]))
fw_best=wr[best_O]

mu_range=np.logspace(-15,0,100)

b1,v1,t1=bias_variance(Kg,P_H1,f_true,s40,mu_range)
b1l,v1l,t1l=bias_variance(Kg,P_H1,f_true,s20,mu_range)

zoom=(40,110,70,190)
r0,r1,c0,c1=zoom

print("Computations done. Generating figures...")

# ========================================================
# Figure 1: Zoomed L2 vs H1 vs H2 at 40 dB
# ========================================================
fig,axes=plt.subplots(2,4,figsize=(16,8))
for i,(img,ttl) in enumerate([(f_true,'Original'),(rh['L2'],f'$L^2$ PSNR={compute_psnr(f_true,rh["L2"]):.1f}'),(rh['H1'],f'$H^1$ PSNR={compute_psnr(f_true,rh["H1"]):.1f}'),(rh['H2'],f'$H^2$ PSNR={compute_psnr(f_true,rh["H2"]):.1f}')]):
    axes[0,i].imshow(img,cmap='gray'); axes[0,i].set_title(ttl,fontsize=11); axes[0,i].axis('off')
    rect=Rectangle((c0,r0),c1-c0,r1-r0,lw=2,edgecolor='red',facecolor='none')
    axes[0,i].add_patch(rect)
    axes[1,i].imshow(img[r0:r1,c0:c1],cmap='gray'); axes[1,i].set_title('Zoomed',fontsize=10); axes[1,i].axis('off')
fig.suptitle('Figure 1: $L^2$, $H^1$, $H^2$ Reconstructions -SNR = 40 dB (Gaussian Blur)',fontsize=13,fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,'fig1_zoom_40dB.png'),dpi=150,bbox_inches='tight'); plt.close()

# ========================================================
# Figure 2: Zoomed L2 vs H1 vs H2 at 20 dB
# ========================================================
fig,axes=plt.subplots(2,4,figsize=(16,8))
for i,(img,ttl) in enumerate([(f_true,'Original'),(rl['L2'],f'$L^2$ PSNR={compute_psnr(f_true,rl["L2"]):.1f}'),(rl['H1'],f'$H^1$ PSNR={compute_psnr(f_true,rl["H1"]):.1f}'),(rl['H2'],f'$H^2$ PSNR={compute_psnr(f_true,rl["H2"]):.1f}')]):
    axes[0,i].imshow(img,cmap='gray'); axes[0,i].set_title(ttl,fontsize=11); axes[0,i].axis('off')
    rect=Rectangle((c0,r0),c1-c0,r1-r0,lw=2,edgecolor='red',facecolor='none')
    axes[0,i].add_patch(rect)
    axes[1,i].imshow(img[r0:r1,c0:c1],cmap='gray'); axes[1,i].set_title('Zoomed',fontsize=10); axes[1,i].axis('off')
fig.suptitle('Figure 2: $L^2$, $H^1$, $H^2$ Reconstructions -SNR = 20 dB (Gaussian Blur)',fontsize=13,fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,'fig2_zoom_20dB.png'),dpi=150,bbox_inches='tight'); plt.close()

# ========================================================
# Figure 3: All blur types x all penalties grid
# ========================================================
fig,axes=plt.subplots(3,4,figsize=(18,13))
for row,(bname,res) in enumerate([('Gaussian',rh),('Motion',rm),('Out-of-Focus',ro)]):
    axes[row,0].imshow(f_true,cmap='gray')
    axes[row,0].set_title('Original' if row==0 else '',fontsize=10)
    axes[row,0].set_ylabel(bname,fontsize=12,fontweight='bold'); axes[row,0].axis('off')
    for col,pn in enumerate(['L2','H1','H2']):
        p=compute_psnr(f_true,res[pn])
        axes[row,col+1].imshow(res[pn],cmap='gray')
        axes[row,col+1].set_title(f'{pn} PSNR={p:.1f}' if row==0 else f'PSNR={p:.1f}',fontsize=10)
        axes[row,col+1].axis('off')
fig.suptitle('Figure 3: All Blur Types x All Penalties (SNR = 40 dB)',fontsize=13,fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,'fig3_allblur_allpen.png'),dpi=150,bbox_inches='tight'); plt.close()

# ========================================================
# Figure 4: Spectral windowing vs Tikhonov (zoom)
# ========================================================
fig,axes=plt.subplots(2,3,figsize=(14,9))
for i,(img,ttl) in enumerate([(f_true,'Original'),(fw_best,f'Window $\\Omega$={best_O}\nPSNR={compute_psnr(f_true,fw_best):.1f}'),(rh['H1'],f'Tikhonov $H^1$\nPSNR={compute_psnr(f_true,rh["H1"]):.1f}')]):
    axes[0,i].imshow(img,cmap='gray'); axes[0,i].set_title(ttl,fontsize=11); axes[0,i].axis('off')
    rect=Rectangle((c0,r0),c1-c0,r1-r0,lw=2,edgecolor='red',facecolor='none')
    axes[0,i].add_patch(rect)
    axes[1,i].imshow(img[r0:r1,c0:c1],cmap='gray'); axes[1,i].set_title('Zoomed',fontsize=10); axes[1,i].axis('off')
fig.suptitle('Figure 4: Spectral Windowing vs Tikhonov $H^1$ (SNR = 40 dB)',fontsize=13,fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,'fig4_window_vs_tikh.png'),dpi=150,bbox_inches='tight'); plt.close()

# ========================================================
# Figure 5: Bias-variance trade-off (40 dB and 20 dB)
# ========================================================
fig,axes=plt.subplots(1,2,figsize=(14,6))
for ax,b,v,t,lab in [(axes[0],b1,v1,t1,'SNR = 40 dB'),(axes[1],b1l,v1l,t1l,'SNR = 20 dB')]:
    ax.loglog(mu_range,b,'b-',lw=2,label='Bias$^2$')
    ax.loglog(mu_range,v,'r-',lw=2,label='Variance')
    ax.loglog(mu_range,t,'k--',lw=2,label='Total Error')
    idx=np.argmin(t)
    ax.axvline(mu_range[idx],color='green',ls=':',lw=1.5,label=f'Optimal $\\mu$={mu_range[idx]:.2e}')
    ax.plot(mu_range[idx],t[idx],'go',ms=8)
    ax.set_xlabel('$\\mu$',fontsize=12); ax.set_ylabel('Error',fontsize=12)
    ax.set_title(f'Bias-Variance ($H^1$, {lab})',fontsize=12)
    ax.legend(fontsize=9); ax.grid(True,which='both',alpha=0.3)
fig.suptitle('Figure 5: Bias-Variance Trade-off',fontsize=13,fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,'fig5_biasvar.png'),dpi=150,bbox_inches='tight'); plt.close()

# ========================================================
# Figure 6: Total error comparison L2 vs H1 vs H2
# ========================================================
fig,ax=plt.subplots(figsize=(9,6))
colors={'L2':'blue','H1':'green','H2':'red'}
for n,Psq in pens.items():
    _,_,t=bias_variance(Kg,Psq,f_true,s40,mu_range)
    ax.loglog(mu_range,t,color=colors[n],lw=2,label=f'{n} Total Error')
    idx=np.argmin(t); ax.plot(mu_range[idx],t[idx],'o',color=colors[n],ms=8)
ax.set_xlabel('$\\mu$',fontsize=12); ax.set_ylabel('Total Error',fontsize=12)
ax.set_title('Figure 6: Total Error -$L^2$ vs $H^1$ vs $H^2$ (SNR = 40 dB)',fontsize=12)
ax.legend(fontsize=11); ax.grid(True,which='both',alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,'fig6_total_err.png'),dpi=150,bbox_inches='tight'); plt.close()

print("Figures saved. Building PDF...")

# ========================================================
# BUILD THE PDF REPORT
# ========================================================
class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica','I',8)
            self.cell(0,10,'Generalized Tikhonov Regularization - Report',align='C',new_x="LMARGIN",new_y="NEXT")
            self.line(10,18,200,18)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica','I',8)
        self.cell(0,10,f'Page {self.page_no()}/{{nb}}',align='C')

    def section_title(self, title):
        self.set_font('Helvetica','B',13)
        self.cell(0,10,title,new_x="LMARGIN",new_y="NEXT")
        self.ln(2)

    def subsection_title(self, title):
        self.set_font('Helvetica','B',11)
        self.cell(0,8,title,new_x="LMARGIN",new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font('Helvetica','',10)
        self.multi_cell(0,5.5,text)
        self.ln(2)

    def bullet(self, text, bold_prefix=""):
        self.set_font('Helvetica','',10)
        x = self.get_x()
        self.cell(5, 5.5, '-')  # bullet
        if bold_prefix:
            self.set_font('Helvetica','B',10)
            self.write(5.5, bold_prefix + " ")
            self.set_font('Helvetica','',10)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def add_figure(self, path, caption="", w=180):
        if os.path.exists(path):
            self.image(path, x=(210-w)/2, w=w)
            if caption:
                self.ln(2)
                self.set_font('Helvetica','I',9)
                self.multi_cell(0,4.5,caption,align='C')
            self.ln(4)


pdf = Report()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# ---- TITLE PAGE ----
pdf.add_page()
pdf.ln(40)
pdf.set_font('Helvetica','B',22)
pdf.cell(0,12,'Generalized Tikhonov Regularization',align='C',new_x="LMARGIN",new_y="NEXT")
pdf.set_font('Helvetica','B',16)
pdf.cell(0,10,'& Error Analysis',align='C',new_x="LMARGIN",new_y="NEXT")
pdf.ln(8)
pdf.set_font('Helvetica','',13)
pdf.cell(0,8,'Report',align='C',new_x="LMARGIN",new_y="NEXT")
pdf.ln(15)
pdf.set_font('Helvetica','',11)
pdf.cell(0,7,'Mathematics for Imaging and Signal Processing',align='C',new_x="LMARGIN",new_y="NEXT")
pdf.cell(0,7,'A.A. 2025/2026',align='C',new_x="LMARGIN",new_y="NEXT")

# ---- SECTION 1 ----
pdf.add_page()
pdf.section_title('1. Visual Comparison of L2, H1, H2 Reconstructions')

pdf.body_text(
    'The Shepp-Logan phantom (256x256) was chosen as the test image because its sharp elliptical '
    'boundaries at various angles make it ideal for evaluating edge preservation under different '
    'regularization penalties. Three blur models were implemented: Gaussian, Linear Motion (with phase '
    'factor), and Out-of-Focus (Bessel J1). Observations were generated at SNR = 40 dB and 20 dB.'
)

pdf.subsection_title('1.1 Gaussian Blur -High SNR (40 dB)')
pdf.body_text(
    'At 40 dB the noise is barely visible. The differences between penalties are subtle but clear '
    'in the zoomed crops (red boxes mark the zoom region):'
)
pdf.bullet('treats all frequencies with the same weight. Edges are uniformly smoothed.', bold_prefix='L2:')
pdf.bullet('penalizes higher frequencies proportionally to |w|^2, so low/mid frequencies '
           '(where edges live) pass through with less damping. Edges are noticeably sharper.', bold_prefix='H1:')
pdf.bullet('the |w|^4 penalty crushes high frequencies very aggressively. The image is very smooth '
           'but fine detail is lost.', bold_prefix='H2:')

pdf.add_figure(os.path.join(FIG_DIR,'fig1_zoom_40dB.png'))

pdf.subsection_title('1.2 Gaussian Blur -Low SNR (20 dB)')
pdf.body_text(
    'At 20 dB the differences are much more dramatic. L2 lets a lot of granular noise through '
    '(visible in flat regions). H1 still keeps edges reasonably sharp while damping most noise. '
    'H2 produces the cleanest image but looks over-smoothed -all fine structure is gone.'
)

pdf.add_figure(os.path.join(FIG_DIR,'fig2_zoom_20dB.png'))

pdf.subsection_title('1.3 Across All Blur Types')
pdf.body_text(
    'The same pattern holds across Gaussian, Motion, and Out-of-Focus blurs. H1 consistently gives '
    'the best balance between sharpness and noise removal. The out-of-focus case is the hardest to '
    'reconstruct because its transfer function has actual zeros (ring-shaped in the Fourier plane), '
    'meaning some frequency content is irrecoverably lost regardless of the penalty chosen.'
)

pdf.add_figure(os.path.join(FIG_DIR,'fig3_allblur_allpen.png'))

pdf.subsection_title('1.4 Spectral Windowing vs Tikhonov')
pdf.body_text(
    'Comparing the hard rectangular cutoff against Tikhonov H1, the windowed reconstruction shows '
    'clear Gibbs ringing (oscillations) near every sharp edge. This is because a rectangular window in '
    'frequency is equivalent to convolving with a sinc in space, and the sinc\'s sidelobes create those '
    'ripples. Tikhonov avoids this entirely through its smooth roll-off.'
)

pdf.add_figure(os.path.join(FIG_DIR,'fig4_window_vs_tikh.png'))

# ---- SECTION 2 ----
pdf.add_page()
pdf.section_title('2. The Bias-Variance Trade-off')

pdf.body_text(
    'The total reconstruction error decomposes into two competing terms:'
)
pdf.body_text(
    '    Total Error^2(mu) = ||(R_mu A - I) f_true||^2  +  ||R_mu w||^2\n'
    '                        [  Bias (detail loss)  ]    [ Variance (noise) ]'
)
pdf.body_text(
    'Both terms were computed in the frequency domain via Parseval\'s identity for the H1 penalty, '
    'sweeping mu from 10^-6 to 10^0.'
)

pdf.add_figure(os.path.join(FIG_DIR,'fig5_biasvar.png'))

pdf.body_text(
    'Key observations from the bias-variance plots:'
)
pdf.bullet('Bias increases monotonically with mu. More regularization means the reconstruction '
           'filter deviates further from the ideal inverse, and we lose more image content.')
pdf.bullet('Variance decreases monotonically with mu. Stronger regularization damps the filter, '
           'so less noise passes through.')
pdf.bullet('The total error is U-shaped on a log-log scale. Its minimum (the optimal mu*) sits '
           'roughly where the bias and variance curves intersect.')
pdf.bullet('Comparing 40 dB vs 20 dB: the bias curve is identical (it only depends on the true '
           'image and blur kernel, not on noise). The variance curve shifts upward proportionally '
           'to sigma^2. This pushes the optimal mu* to a larger value -noisier data needs more '
           'regularization.')

pdf.subsection_title('2.1 Comparing L2, H1, H2 Total Error')
pdf.body_text(
    'Plotting the total error curves for all three penalties shows that H1 achieves the lowest '
    'minimum total error for this test image. This makes sense: the Shepp-Logan phantom is piecewise '
    'constant with sharp edges, and H1 (gradient penalty) is designed to preserve exactly that kind '
    'of structure.'
)

pdf.add_figure(os.path.join(FIG_DIR,'fig6_total_err.png'))

# ---- SECTION 3 ----
pdf.add_page()
pdf.section_title('3. Effect of Smoothness Order on Edge Preservation and Noise Suppression')

pdf.body_text(
    'The key difference between H1 and H2 comes down to how fast the penalty grows with frequency:'
)

pdf.subsection_title('H1 Penalty: |P|^2 = |w|^2')
pdf.bullet('DC (w = 0) is not penalized at all -the mean brightness is preserved exactly.')
pdf.bullet('Low frequencies get a small penalty -the large-scale structure comes through.')
pdf.bullet('High frequencies get a moderate penalty -noise is damped, but edges (which need '
           'mid-to-high frequencies) are still partially retained.')
pdf.bullet('Result: good compromise -edges are visible and noise is controlled.')

pdf.subsection_title('H2 Penalty: |P|^2 = |w|^4')
pdf.bullet('Low frequencies are barely penalized (similar to H1).')
pdf.bullet('Mid-to-high frequencies are crushed much more heavily than in H1.')
pdf.bullet('Edge detail, which lives in the mid-to-high frequency range, gets suppressed '
           'along with the noise.')
pdf.bullet('Result: very smooth -noise is almost gone, but the image looks blurry and fine '
           'detail is lost.')

pdf.ln(3)
pdf.body_text(
    'In short: increasing the smoothness order from H1 to H2 improves noise suppression but hurts '
    'edge preservation. For images with sharp boundaries (like medical or scientific images), H1 is '
    'usually the better choice. H2 might be preferable only when noise is very strong and we are '
    'willing to sacrifice detail for a cleaner result.'
)

pdf.body_text(
    'The spectral windowing comparison reinforces these findings: the hard rectangular cutoff avoids '
    'the smooth penalty altogether and just chops frequencies at a threshold. This is conceptually '
    'simpler but produces Gibbs ringing near edges. Tikhonov with any of the three penalties avoids '
    'this because it uses a smooth roll-off instead of a sharp cutoff.'
)

pdf.ln(5)
pdf.subsection_title('Summary Table')

# Summary table
pdf.set_font('Helvetica','B',10)
col_w = [25, 50, 50, 65]
headers = ['Penalty', 'Edge Preservation', 'Noise Suppression', 'Best Use Case']
for i,h in enumerate(headers):
    pdf.cell(col_w[i],7,h,border=1,align='C')
pdf.ln()

pdf.set_font('Helvetica','',9)
rows = [
    ['L2', 'Worst (uniform blur)', 'Moderate', 'Simple regularization baseline'],
    ['H1', 'Good (edges sharp)', 'Good', 'Piecewise-constant images, edges'],
    ['H2', 'Over-smoothed', 'Best', 'Very noisy data, smooth targets'],
    ['Window', 'Gibbs ringing', 'Depends on cutoff', 'Quick & dirty, no mu tuning'],
]
for row in rows:
    for i,val in enumerate(row):
        pdf.cell(col_w[i],6,val,border=1)
    pdf.ln()


# ---- SAVE ----
out_path = os.path.join(OUT_DIR, 'report.pdf')
pdf.output(out_path)
print(f"\nReport saved to: {out_path}")

# Cleanup figures
import shutil
shutil.rmtree(FIG_DIR)
print("Temp figures cleaned up.")
