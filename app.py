import cv2
import numpy as np
import streamlit as st
from ultralytics import SAM
import os

st.set_page_config(layout="centered")
st.title("🦎 Vuursalamander AI Monitor (SAM + SIFT)")
st.write("Upload of maak een foto. De AI isoleert het dier en vergelijkt het patroon met de database.")

# Map aanmaken voor de visuele database (uitgesneden salamanders)
DB_DIR = "database_salamanders"
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)


# Laad SAM eenmalig in het geheugen zodat de app snel blijft
@st.cache_resource
def laad_sam():
    return SAM("mobile_sam.pt")


sam_model = laad_sam()


def isoleer_salamander(img):
    h_orig, w_orig, _ = img.shape
    schaal = 800.0 / float(max(h_orig, w_orig))
    nieuwe_w = int(w_orig * schaal)
    nieuwe_h = int(h_orig * schaal)
    img_resized = cv2.resize(img, (nieuwe_w, nieuwe_h))

    box = [[int(nieuwe_w * 0.1), int(nieuwe_h * 0.1), int(nieuwe_w * 0.9), int(nieuwe_h * 0.9)]]
    results = sam_model(img_resized, bboxes=box, verbose=False)

    if len(results) > 0 and results[0].masks is not None:
        masker = results[0].masks.data[0].cpu().numpy()
        masker = (masker * 255).astype(np.uint8)
        masker = cv2.resize(masker, (nieuwe_w, nieuwe_h))
        return cv2.bitwise_and(img_resized, img_resized, mask=masker)
    return img_resized


def vergelijk_patronen(sal_nieuw, sal_db):
    grijs_1 = cv2.cvtColor(sal_nieuw, cv2.COLOR_BGR2GRAY)
    grijs_2 = cv2.cvtColor(sal_db, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(grijs_1, None)
    kp2, des2 = sift.detectAndCompute(grijs_2, None)

    if des1 is None or des2 is None:
        return 0, None

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    goede_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.70 * n.distance:
                goede_matches.append([m])

    match_img = cv2.drawMatchesKnn(
        sal_nieuw, kp1, sal_db, kp2, goede_matches, outImg=np.array([]),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return len(goede_matches), match_img


# --- DE WEB INTERFACE ---
uploaded_file = st.file_uploader("Fotografeer of kies een salamander...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with st.spinner("🤖 AI snijdt de salamander los van de achtergrond..."):
        geisoleerde_sal = isoleer_salamander(image)

    st.image(cv2.cvtColor(geisoleerde_sal, cv2.COLOR_BGR2RGB), caption="Geïsoleerd individu", width="stretch")

    # Vergelijk met alle opgeslagen salamanders in de map
    beste_match_naam = None
    hoogste_aantal_matches = 0
    beste_match_foto = None

    opgeslagen_bestanden = [f for f in os.listdir(DB_DIR) if f.endswith(".png")]

    if len(opgeslagen_bestanden) > 0:
        with st.spinner(f"🔬 Vergelijken met {len(opgeslagen_bestanden)} bekende individuen..."):
            for bestand in opgeslagen_bestanden:
                db_pad = os.path.join(DB_DIR, bestand)
                db_img = cv2.imread(db_pad)
                aantal, m_img = vergelijk_patronen(geisoleerde_sal, db_img)

                if aantal > hoogste_aantal_matches:
                    hoogste_aantal_matches = aantal
                    beste_match_naam = os.path.splitext(bestand)[0]
                    beste_match_foto = m_img

    # Drempelwaarde voor herkenning: 20 overeenkomstige ankerpunten
    if beste_match_naam and hoogste_aantal_matches >= 20:
        st.success(f"🎉 BEKEND INDIVIDU HERKEND: **{beste_match_naam}**!")
        st.write(f"Aantal overeenkomstige SIFT-ankerpunten: **{hoogste_aantal_matches}**")
        st.image(cv2.cvtColor(beste_match_foto, cv2.COLOR_BGR2RGB), caption="Visueel bewijs van de match",
                 width="stretch")
    else:
        st.warning(f"⚠️ NIEUW INDIVIDU! (Hoogste gelijkenis met database was slechts {hoogste_aantal_matches} punten)")

        standaard_naam = f"Salamander_{len(opgeslagen_bestanden) + 1:03d}"
        aangepaste_naam = st.text_input("Geef deze salamander een unieke naam/code:", value=standaard_naam)

        if st.button(f"Opslaan als {aangepaste_naam}"):
            # Maak bestandsnaam veilig en sla de uitgesneden foto op
            veilige_naam = "".join([c for c in aangepaste_naam if c.isalnum() or c in (' ', '-', '_')]).strip()
            opslaan_pad = os.path.join(DB_DIR, f"{veilige_naam}.png")

            if os.path.exists(opslaan_pad):
                st.error("Deze naam bestaat al in de database! Kies een andere naam.")
            else:
                cv2.imwrite(opslaan_pad, geisoleerde_sal)
                st.info(f"**{veilige_naam}** is toegevoegd aan de database!")
                st.rerun()

# Overzicht onderaan de pagina
st.divider()
huidige_db = [os.path.splitext(f)[0] for f in os.listdir(DB_DIR) if f.endswith(".png")]
st.subheader(f"📁 Huidige Populatie Database ({len(huidige_db)} individuen)")
if len(huidige_db) > 0:
    st.write(", ".join(huidige_db))
else:
    st.write("De database is nog leeg.")
