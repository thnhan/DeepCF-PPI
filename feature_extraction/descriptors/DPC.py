"""
The dipeptide composition is
used to transform the variable length of proteins to fixed length
feature vectors. A dipeptide composition has been used earlier by
Grassmann et al.67 and Reczko and Bohr68 for the development
of fold recognition methods. We adopt the same dipeptide
composition-based approach in developing a deep neural
Networks-based method for predicting protein−protein interaction.
The dipeptide composition gives a fixed pattern length of
400. Dipeptide composition encapsulates information about the
fraction of amino acids as well as their local order. The dipeptide
composition is defined as:

@coding: thnhan
"""
from collections import Counter

di_peptides = ['AA', 'AR', 'AN', 'AD', 'AC', 'AE', 'AQ', 'AG', 'AH', 'AI', 'AL', 'AK', 'AM', 'AF', 'AP', 'AS', 'AT',
               'AW', 'AY', 'AV', 'RA', 'RR', 'RN', 'RD', 'RC', 'RE', 'RQ', 'RG', 'RH', 'RI', 'RL', 'RK', 'RM', 'RF',
               'RP', 'RS', 'RT', 'RW', 'RY', 'RV', 'NA', 'NR', 'NN', 'ND', 'NC', 'NE', 'NQ', 'NG', 'NH', 'NI', 'NL',
               'NK', 'NM', 'NF', 'NP', 'NS', 'NT', 'NW', 'NY', 'NV', 'DA', 'DR', 'DN', 'DD', 'DC', 'DE', 'DQ', 'DG',
               'DH', 'DI', 'DL', 'DK', 'DM', 'DF', 'DP', 'DS', 'DT', 'DW', 'DY', 'DV', 'CA', 'CR', 'CN', 'CD', 'CC',
               'CE', 'CQ', 'CG', 'CH', 'CI', 'CL', 'CK', 'CM', 'CF', 'CP', 'CS', 'CT', 'CW', 'CY', 'CV', 'EA', 'ER',
               'EN', 'ED', 'EC', 'EE', 'EQ', 'EG', 'EH', 'EI', 'EL', 'EK', 'EM', 'EF', 'EP', 'ES', 'ET', 'EW', 'EY',
               'EV', 'QA', 'QR', 'QN', 'QD', 'QC', 'QE', 'QQ', 'QG', 'QH', 'QI', 'QL', 'QK', 'QM', 'QF', 'QP', 'QS',
               'QT', 'QW', 'QY', 'QV', 'GA', 'GR', 'GN', 'GD', 'GC', 'GE', 'GQ', 'GG', 'GH', 'GI', 'GL', 'GK', 'GM',
               'GF', 'GP', 'GS', 'GT', 'GW', 'GY', 'GV', 'HA', 'HR', 'HN', 'HD', 'HC', 'HE', 'HQ', 'HG', 'HH', 'HI',
               'HL', 'HK', 'HM', 'HF', 'HP', 'HS', 'HT', 'HW', 'HY', 'HV', 'IA', 'IR', 'IN', 'ID', 'IC', 'IE', 'IQ',
               'IG', 'IH', 'II', 'IL', 'IK', 'IM', 'IF', 'IP', 'IS', 'IT', 'IW', 'IY', 'IV', 'LA', 'LR', 'LN', 'LD',
               'LC', 'LE', 'LQ', 'LG', 'LH', 'LI', 'LL', 'LK', 'LM', 'LF', 'LP', 'LS', 'LT', 'LW', 'LY', 'LV', 'KA',
               'KR', 'KN', 'KD', 'KC', 'KE', 'KQ', 'KG', 'KH', 'KI', 'KL', 'KK', 'KM', 'KF', 'KP', 'KS', 'KT', 'KW',
               'KY', 'KV', 'MA', 'MR', 'MN', 'MD', 'MC', 'ME', 'MQ', 'MG', 'MH', 'MI', 'ML', 'MK', 'MM', 'MF', 'MP',
               'MS', 'MT', 'MW', 'MY', 'MV', 'FA', 'FR', 'FN', 'FD', 'FC', 'FE', 'FQ', 'FG', 'FH', 'FI', 'FL', 'FK',
               'FM', 'FF', 'FP', 'FS', 'FT', 'FW', 'FY', 'FV', 'PA', 'PR', 'PN', 'PD', 'PC', 'PE', 'PQ', 'PG', 'PH',
               'PI', 'PL', 'PK', 'PM', 'PF', 'PP', 'PS', 'PT', 'PW', 'PY', 'PV', 'SA', 'SR', 'SN', 'SD', 'SC', 'SE',
               'SQ', 'SG', 'SH', 'SI', 'SL', 'SK', 'SM', 'SF', 'SP', 'SS', 'ST', 'SW', 'SY', 'SV', 'TA', 'TR', 'TN',
               'TD', 'TC', 'TE', 'TQ', 'TG', 'TH', 'TI', 'TL', 'TK', 'TM', 'TF', 'TP', 'TS', 'TT', 'TW', 'TY', 'TV',
               'WA', 'WR', 'WN', 'WD', 'WC', 'WE', 'WQ', 'WG', 'WH', 'WI', 'WL', 'WK', 'WM', 'WF', 'WP', 'WS', 'WT',
               'WW', 'WY', 'WV', 'YA', 'YR', 'YN', 'YD', 'YC', 'YE', 'YQ', 'YG', 'YH', 'YI', 'YL', 'YK', 'YM', 'YF',
               'YP', 'YS', 'YT', 'YW', 'YY', 'YV', 'VA', 'VR', 'VN', 'VD', 'VC', 'VE', 'VQ', 'VG', 'VH', 'VI', 'VL',
               'VK', 'VM', 'VF', 'VP', 'VS', 'VT', 'VW', 'VY', 'VV']


class DPCEncoder:
    def __init__(self):
        self.minLength = 2  # Length conditions
        self.dim = 400
        self.shortName = 'DC'
        self.fullName = 'DiPeptide Composition'

    def to_feature(self, sequence):
        if len(sequence) >= self.minLength:
            L = len(sequence) - 1
            n_di = [sequence[i:i + 2] for i in range(L)]
            n_di = Counter(n_di)
            features = [n_di[di] / L for di in di_peptides]
            return features
        else:
            print("Error length")


if __name__ == "__main__":
    seq = 'AFQVNTNINAMNAHVQSALTQNALKTSLERLSSGLRINKAADDASGMTVADSLRSQASSLGQAIANTNDGMGIIQVADKAMDEQLKILDTVKVKA' \
          'TQAAQDGQTTESRKAIQSDIVRLIQGLDNIGNTTTYNGQALLSGQFTNKEFQVGAYSNQSIKASIGSTTSDKIGQVRIATGALITASGDISLTFK' \
          'QVDGVNDVTLESVKVSSSAGTGIGVLAEVINKNSNRTGVKAYASVITTSDVAVQSGSLSNLTLNGIHLGNIADIKKNDSDGRLVAAINAVTSETG' \
          'VEAYTDQKGRLNLRSIDGRGIEIKTDSVSNGPSALTMVNGGQDLTKGSTNYGRLSLTRLDAKSINVVSASDSQHLGFTAIGFGESQVAETTVNLR' \
          'DVTGNFNANVKSASGANYNAVIASGNQSLGSGVTTLRGAMVVIDIAESAMKMLDKVRSDLGSVQNQMISTVNNISITQVNVKAAESQIRDVDFAE' \
          'ESANFNKNNILAQSGSYAMSQANTVQQNILRLLT'
    feat = DPCEncoder().to_feature(seq)
    print(feat)
